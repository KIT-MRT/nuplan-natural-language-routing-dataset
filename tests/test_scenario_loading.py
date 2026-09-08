import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from route_description_generation import scenario_loading as sl
from route_description_generation.scenario_loading import (
    SIMULATION_EXTRACTION_OFFSET_S,
    FixedWindowScenarioMapping,
    resolve_data_path,
    resolve_extraction_offset,
    resolve_log_names,
    resolve_maps_path,
    resolve_scenario_tokens_file,
)


class TestRoutingSplitResolution(unittest.TestCase):
    def test_resolve_log_names_prefers_explicit_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            explicit = root / "explicit.json"
            explicit.write_text('["log_a.db"]', encoding="utf-8")
            split_dir = root / "splits"
            split_dir.mkdir()
            (split_dir / "nuplan_train.json").write_text('["log_b.db"]', encoding="utf-8")

            result = resolve_log_names("train", explicit, split_dir)
            self.assertEqual(result, ["log_a.db"])

    def test_resolve_log_names_uses_split_file_when_present(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            split_dir = root / "splits"
            split_dir.mkdir()
            (split_dir / "nuplan_val.json").write_text(
                '["val_log_1.db", "val_log_2.db"]', encoding="utf-8"
            )

            result = resolve_log_names("val", None, split_dir)
            self.assertEqual(result, ["val_log_1.db", "val_log_2.db"])

    def test_resolve_log_names_returns_none_if_split_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_dir = Path(tmp_dir)
            result = resolve_log_names("interplan", None, split_dir)
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()


class TestResolveDataPath(unittest.TestCase):
    ENV = {
        "NUPLAN_DATA_ROOT": "/data/nuplan",
        "NUPLAN_MAPS_ROOT": "/data/nuplan/maps",
    }

    def test_train_val_and_val14_resolve_to_trainval(self):
        with mock.patch.dict(os.environ, self.ENV, clear=True):
            for split in ("train", "val", "val14"):
                with self.subTest(split=split):
                    self.assertEqual(
                        resolve_data_path(split),
                        Path("/data/nuplan/nuplan-v1.1/splits/trainval"),
                    )

    def test_any_other_split_resolves_to_test(self):
        with mock.patch.dict(os.environ, self.ENV, clear=True):
            self.assertEqual(
                resolve_data_path("interplan"),
                Path("/data/nuplan/nuplan-v1.1/splits/test"),
            )

    def test_returns_none_without_a_split(self):
        with mock.patch.dict(os.environ, self.ENV, clear=True):
            self.assertIsNone(resolve_data_path(None))

    def test_returns_none_when_the_environment_is_silent(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_data_path("val14"))
            self.assertIsNone(resolve_maps_path())

    def test_maps_path_comes_from_the_environment(self):
        with mock.patch.dict(os.environ, self.ENV, clear=True):
            self.assertEqual(resolve_maps_path(), Path("/data/nuplan/maps"))


class TestResolveScenarioTokensFile(unittest.TestCase):
    def test_returns_the_shipped_token_list_for_the_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            res_dir = Path(tmp)
            (res_dir / "val14_tokens.txt").write_text("abc\n")
            self.assertEqual(
                resolve_scenario_tokens_file("val14", res_dir),
                res_dir / "val14_tokens.txt",
            )

    def test_returns_none_when_the_split_ships_no_token_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(resolve_scenario_tokens_file("val", Path(tmp)))

    def test_returns_none_without_a_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(resolve_scenario_tokens_file(None, Path(tmp)))

    def test_the_repository_ships_a_token_list_for_val14(self):
        """val14 is defined by its token list; losing it would silently widen the split."""
        res_dir = Path(__file__).resolve().parents[1] / "res"
        self.assertIsNotNone(resolve_scenario_tokens_file("val14", res_dir))


class TestBatchTokens(unittest.TestCase):
    def test_no_filter_becomes_a_single_unfiltered_query(self):
        self.assertEqual(sl._batch_tokens(None, 10), [None])

    def test_a_short_list_stays_one_batch(self):
        self.assertEqual(sl._batch_tokens(["a", "b"], 10), [["a", "b"]])

    def test_a_long_list_is_split_without_losing_or_duplicating_tokens(self):
        tokens = [str(i) for i in range(25)]
        batches = sl._batch_tokens(tokens, 10)
        self.assertEqual([len(b) for b in batches], [10, 10, 5])
        flattened = [t for batch in batches for t in batch]
        self.assertEqual(flattened, tokens)


class TestLargeTokenFilter(unittest.TestCase):
    """nuPlan-devkit binds one SQL variable per token; past SQLite's limit the query fails."""

    def _build(self, scenario_tokens, total_scenarios=None, available=("a", "b", "c", "d")):
        by_token = {t: mock.Mock(token=t) for t in available}
        calls = []

        class _FakeBuilder:
            def __init__(self, *args, **kwargs):
                pass

            def get_scenarios(self, scenario_filter, worker):
                calls.append(scenario_filter)
                requested = scenario_filter[1]
                if requested is None:
                    return list(by_token.values())
                # Mirror the devkit: return exactly the requested tokens that exist.
                return [by_token[t] for t in requested if t in by_token]

        with (
            mock.patch.object(sl, "NuPlanScenarioBuilder", _FakeBuilder),
            mock.patch.object(sl, "SingleMachineParallelExecutor", mock.Mock()),
            mock.patch.object(sl, "ScenarioFilter", lambda *a: a),
        ):
            result = sl.build_scenarios(
                data_path="/d",
                map_path="/m",
                map_version="v",
                log_names=None,
                scenario_tokens=scenario_tokens,
                total_scenarios=total_scenarios,
                scenarios_per_type=None,
                shuffle_scenarios=False,
            )
        return result, calls

    def test_small_filter_is_a_single_query_with_all_tokens(self):
        _, calls = self._build(["a", "b"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], ["a", "b"])

    def test_oversized_filter_is_split_across_queries(self):
        with mock.patch.object(sl, "max_db_filter_tokens", return_value=2):
            result, calls = self._build(["a", "b", "c", "d"])
        self.assertEqual(len(calls), 2)
        self.assertEqual([c[1] for c in calls], [["a", "b"], ["c", "d"]])
        self.assertEqual([s.token for s in result], ["a", "b", "c", "d"])

    def test_batched_result_matches_the_requested_tokens_exactly(self):
        requested = ["a", "b", "c", "d"]
        with mock.patch.object(sl, "max_db_filter_tokens", return_value=1):
            result, _ = self._build(requested)
        self.assertEqual({s.token for s in result}, set(requested))

    def test_tokens_absent_from_the_split_are_reported_not_silently_dropped(self):
        with mock.patch.object(sl, "max_db_filter_tokens", return_value=2):
            result, _ = self._build(["a", "b", "missing"])
        self.assertEqual([s.token for s in result], ["a", "b"])

    def test_a_token_in_two_batches_is_not_returned_twice(self):
        with mock.patch.object(sl, "max_db_filter_tokens", return_value=2):
            result, _ = self._build(["a", "b", "a", "c"])
        self.assertEqual([s.token for s in result], ["a", "b", "c"])

    def test_total_scenarios_is_applied_once_over_the_combined_result(self):
        """Per batch the cap would be applied repeatedly and over-deliver."""
        with mock.patch.object(sl, "max_db_filter_tokens", return_value=2):
            result, calls = self._build(["a", "b", "c", "d"], total_scenarios=3)
        self.assertTrue(all(c[5] is None for c in calls), "the devkit must not also cap")
        self.assertEqual([s.token for s in result], ["a", "b", "c"])

    def test_limit_tracks_the_running_sqlite_version(self):
        self.assertGreater(sl.max_db_filter_tokens(), 900)
        self.assertLess(sl.max_db_filter_tokens(), 32766)


class TestExtractionOffset(unittest.TestCase):
    """The simulation starts the ego 3 s before the tagged token; the dataset must agree."""

    def test_simulation_splits_get_the_simulation_offset(self):
        for split in ("val", "val14", "interplan"):
            with self.subTest(split=split):
                self.assertEqual(resolve_extraction_offset(split), SIMULATION_EXTRACTION_OFFSET_S)

    def test_training_splits_stay_anchored_at_the_token(self):
        self.assertEqual(resolve_extraction_offset("train"), 0.0)
        self.assertEqual(resolve_extraction_offset(None), 0.0)

    def test_the_offset_matches_nuplans_simulation_builders(self):
        """nuplan_challenge/nuplan_eval extract every scenario type at [15.0, -3.0]."""
        self.assertEqual(SIMULATION_EXTRACTION_OFFSET_S, -3.0)

    def test_one_window_is_returned_for_every_scenario_type(self):
        mapping = FixedWindowScenarioMapping(-3.0)
        types = ["starting_left_turn", "stopped_at_traffic_light", "unknown", ""]
        infos = [mapping.get_extraction_info(t) for t in types]

        self.assertEqual(len({id(info) for info in infos}), 1)
        self.assertEqual(infos[0].extraction_offset, -3.0)
        # Subsampling must not move index 0 off the window's first token; see the class docstring.
        self.assertEqual(infos[0].subsample_ratio, 1.0)
        self.assertGreater(infos[0].scenario_duration, 0.0)


class TestScenarioMappingWiring(unittest.TestCase):
    """The offset only reaches the devkit through the builder's scenario_mapping."""

    # Index of expand_scenarios in the positional tuple get_filter_parameters returns.
    _EXPAND_SCENARIOS = 8

    def _build(self, **kwargs):
        """Return ``(builder_kwargs, scenario_filters)`` for one build_scenarios call."""
        captured = {}
        filters = []

        class _FakeBuilder:
            def __init__(self, *args, **builder_kwargs):
                captured.update(builder_kwargs)

            def get_scenarios(self, scenario_filter, worker):
                filters.append(scenario_filter)
                return []

        with (
            mock.patch.object(sl, "NuPlanScenarioBuilder", _FakeBuilder),
            mock.patch.object(sl, "SingleMachineParallelExecutor", mock.Mock()),
            mock.patch.object(sl, "ScenarioFilter", lambda *a: a),
        ):
            sl.build_scenarios(
                data_path="/d",
                map_path="/m",
                map_version="v",
                log_names=None,
                scenario_tokens=None,
                total_scenarios=None,
                scenarios_per_type=None,
                shuffle_scenarios=False,
                **kwargs,
            )
        return captured, filters

    def test_zero_offset_leaves_the_devkit_default_in_place(self):
        """Anything else would risk changing output for runs that never asked for an offset."""
        captured, _ = self._build(extraction_offset=0.0)
        self.assertIsNone(captured["scenario_mapping"])

    def test_default_is_zero_so_existing_callers_are_unaffected(self):
        captured, filters = self._build()
        self.assertIsNone(captured["scenario_mapping"])
        self.assertTrue(filters[0][self._EXPAND_SCENARIOS])

    def test_nonzero_offset_installs_a_fixed_window_mapping(self):
        captured, _ = self._build(extraction_offset=-3.0)
        mapping = captured["scenario_mapping"]
        self.assertIsInstance(mapping, FixedWindowScenarioMapping)
        self.assertEqual(mapping.get_extraction_info("anything").extraction_offset, -3.0)

    def test_an_offset_run_turns_expand_scenarios_off(self):
        """The devkit ignores the mapping entirely while expand_scenarios is set.

        `nuplan_scenario_filter_utils.py:191` passes `scenario_extraction_info=None` whenever
        expand_scenarios is true, collapsing every scenario to the single anchor frame - so leaving
        it on makes the offset a silent no-op rather than an error.
        """
        _, filters = self._build(extraction_offset=-3.0)
        self.assertFalse(filters[0][self._EXPAND_SCENARIOS])

    def test_an_unoffset_run_leaves_expand_scenarios_on(self):
        _, filters = self._build(extraction_offset=0.0)
        self.assertTrue(filters[0][self._EXPAND_SCENARIOS])
