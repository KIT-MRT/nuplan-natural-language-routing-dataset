import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from route_description_generation import dataset_builder as db


class TestChunked(unittest.TestCase):
    def test_splits_into_chunks_of_at_most_the_given_size(self):
        self.assertEqual(list(db.chunked([1, 2, 3, 4, 5], 2)), [[1, 2], [3, 4], [5]])

    def test_empty_input_yields_nothing(self):
        self.assertEqual(list(db.chunked([], 3)), [])


class TestDatasetPaths(unittest.TestCase):
    def test_split_is_part_of_the_file_stem(self):
        paths = db.dataset_paths(Path("/out"), "val14", "lg_routing")
        self.assertEqual(paths.data_file, Path("/out/val14_lg_routing_data.jsonl"))
        self.assertEqual(paths.index_file, Path("/out/val14_lg_routing_index.sqlite"))

    def test_split_is_optional(self):
        paths = db.dataset_paths(Path("/out"), None, "lg_routing")
        self.assertEqual(paths.data_file, Path("/out/lg_routing_data.jsonl"))


class TestResolveInterplanContext(unittest.TestCase):
    def test_non_interplan_split_is_passed_through_untouched(self):
        tokens, goals = db.resolve_interplan_context("val14", False, ["a", "b"])
        self.assertEqual(tokens, ["a", "b"])
        self.assertIsNone(goals)

    def test_interplan_split_loads_goal_variants(self):
        with (
            mock.patch.object(
                db, "resolve_interplan_yaml_paths", return_value=(Path("b.yaml"), Path("m.yaml"))
            ),
            mock.patch.object(db, "load_interplan_tokens", return_value=["x"]),
            mock.patch.object(
                db, "load_interplan_modification_goals", return_value={"x": object()}
            ) as m_goals,
        ):
            tokens, goals = db.resolve_interplan_context("interplan", False, None)

        self.assertEqual(m_goals.call_count, 1)
        self.assertEqual(tokens, ["x"])
        self.assertEqual(list(goals), ["x"])

    def test_explicit_token_filter_is_intersected_with_interplan_tokens(self):
        with (
            mock.patch.object(
                db, "resolve_interplan_yaml_paths", return_value=(Path("b.yaml"), Path("m.yaml"))
            ),
            mock.patch.object(db, "load_interplan_tokens", return_value=["b", "c", "d"]),
            mock.patch.object(db, "load_interplan_modification_goals", return_value={}),
        ):
            tokens, _ = db.resolve_interplan_context("interplan", False, ["a", "b", "c"])

        self.assertEqual(tokens, ["b", "c"])

    def test_variants_without_a_modifications_yaml_is_an_error(self):
        with mock.patch.object(db, "resolve_interplan_yaml_paths", return_value=(None, None)):
            with self.assertRaisesRegex(ValueError, "interplan-variants"):
                db.resolve_interplan_context("val14", True, None)


class TestDatasetWriter(unittest.TestCase):
    def test_rows_are_retrievable_by_token_through_the_index(self):
        from route_description_generation.dataset_index import load_by_token

        with tempfile.TemporaryDirectory() as tmp:
            paths = db.dataset_paths(Path(tmp), "val14", "lg_routing")
            with db._DatasetWriter(paths) as writer:
                writer.write({"token": "first", "payload": 1})
                writer.write({"token": "second", "payload": 2})

            self.assertEqual(
                load_by_token(str(paths.index_file), str(paths.data_file), "second"),
                {"token": "second", "payload": 2},
            )
            self.assertIsNone(load_by_token(str(paths.index_file), str(paths.data_file), "absent"))


if __name__ == "__main__":
    unittest.main()


class TestHasRouteDescription(unittest.TestCase):
    def test_a_normal_row_has_one(self):
        self.assertTrue(db.has_route_description({"routing_data": {"route_description": "Depart"}}))

    def test_empty_and_whitespace_do_not_count(self):
        self.assertFalse(db.has_route_description({"routing_data": {"route_description": ""}}))
        self.assertFalse(db.has_route_description({"routing_data": {"route_description": "  "}}))

    def test_missing_fields_do_not_count(self):
        self.assertFalse(db.has_route_description({"routing_data": {}}))
        self.assertFalse(db.has_route_description({}))


class TestRouteScenariosTally(unittest.TestCase):
    """A scenario that raises writes no row, so the summary has to account for it."""

    def _run(self, outcomes):
        from concurrent.futures import Future

        class _FakeExecutor:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def submit(self, _fn, scenario):
                future = Future()
                outcome = outcomes[scenario.token]
                if isinstance(outcome, Exception):
                    future.set_exception(outcome)
                else:
                    future.set_result(outcome)
                return future

        scenarios = [mock.Mock(token=t) for t in outcomes]
        writer = mock.Mock()
        writer.paths = db.DatasetPaths(Path("d.jsonl"), Path("i.sqlite"))

        with mock.patch.object(db, "ProcessPoolExecutor", _FakeExecutor):
            return db.route_scenarios(
                scenarios, writer, workers=1, chunk_size=10, worker_settings=()
            )

    @staticmethod
    def _row(token, valid=True, description="Depart"):
        return {
            "token": token,
            "routing_data": {"valid_route": valid, "route_description": description},
        }

    def test_counts_written_valid_failed_and_missing_descriptions(self):
        summary = self._run(
            {
                "ok": self._row("ok"),
                "invalid": self._row("invalid", valid=False),
                "nodesc": self._row("nodesc", valid=False, description=""),
                "boom": RuntimeError("no future trajectory"),
            }
        )
        self.assertEqual(summary.processed, 4)
        self.assertEqual(summary.written, 3, "the raising scenario writes no row")
        self.assertEqual(summary.valid, 1)
        self.assertEqual(summary.failed, 1)
        self.assertEqual(summary.without_description, 1)

    def test_processed_is_written_plus_failed(self):
        summary = self._run({"a": self._row("a"), "b": RuntimeError("x"), "c": RuntimeError("y")})
        self.assertEqual(summary.processed, summary.written + summary.failed)
        self.assertEqual(summary.failed, 2)


class TestCollectTokensPrefilter(unittest.TestCase):
    """collect_tokens skips rows by substring first; the fast path must never disagree."""

    def _write(self, tmp, lines):
        path = Path(tmp) / "rows.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_default_spacing_is_matched(self):
        from route_description_generation.dataset_index import collect_tokens

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                [
                    json.dumps({"token": "a", "routing_data": {"valid_route": True}}),
                    json.dumps({"token": "b", "routing_data": {"valid_route": False}}),
                ],
            )
            self.assertEqual(collect_tokens(path, valid_route=False), ["b"])
            self.assertEqual(collect_tokens(path, valid_route=True), ["a"])

    def test_compact_spacing_is_matched_too(self):
        """A file written with compact separators must not silently yield nothing."""
        from route_description_generation.dataset_index import collect_tokens

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                [
                    json.dumps(
                        {"token": "a", "routing_data": {"valid_route": True}},
                        separators=(",", ":"),
                    ),
                    json.dumps(
                        {"token": "b", "routing_data": {"valid_route": False}},
                        separators=(",", ":"),
                    ),
                ],
            )
            self.assertEqual(collect_tokens(path, valid_route=False), ["b"])

    def test_other_false_fields_do_not_produce_false_matches(self):
        """route_validation carries its own booleans; only valid_route decides."""
        from route_description_generation.dataset_index import collect_tokens

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                [
                    json.dumps(
                        {
                            "token": "a",
                            "routing_data": {
                                "valid_route": True,
                                "route_validation": {"length_valid": False},
                            },
                        }
                    ),
                    json.dumps({"token": "b", "routing_data": {"valid_route": False}}),
                ],
            )
            self.assertEqual(collect_tokens(path, valid_route=False), ["b"])

    def test_rows_without_the_field_match_neither(self):
        from route_description_generation.dataset_index import collect_tokens

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [json.dumps({"token": "a", "routing_data": {}})])
            self.assertEqual(collect_tokens(path, valid_route=False), [])
            self.assertEqual(collect_tokens(path, valid_route=True), [])

    def test_agrees_with_an_unfiltered_scan(self):
        from route_description_generation.dataset_index import collect_tokens, iter_rows

        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                {"token": f"t{i}", "routing_data": {"valid_route": i % 3 == 0}} for i in range(50)
            ]
            path = self._write(tmp, [json.dumps(r) for r in rows])
            expected = [
                r["token"] for r in iter_rows(path) if r["routing_data"]["valid_route"] is False
            ]
            self.assertEqual(collect_tokens(path, valid_route=False), expected)
