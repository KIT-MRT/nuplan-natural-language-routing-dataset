import json
import tempfile
import unittest
from pathlib import Path

from route_description_generation.cli.prune_npz import (
    BACKUP_SUFFIX,
    INVALID_SUBDIR,
    SURPLUS_SUBDIR,
    find_scenario_lists,
    index_npz_by_token,
    prune_npz,
)


def _write_dataset(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as out:
        for token, valid in rows:
            payload = {"token": token, "routing_data": {"valid_route": valid}}
            out.write(json.dumps(payload) + "\n")


class TestIndexNpzByToken(unittest.TestCase):
    def test_indexes_by_the_token_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir = Path(tmp)
            (npz_dir / "sg-one-north_aaa.npz").touch()
            (npz_dir / "us-ma-boston_bbb.npz").touch()
            (npz_dir / "notes.txt").touch()
            self.assertEqual(sorted(index_npz_by_token(npz_dir)), ["aaa", "bbb"])

    def test_ignores_files_already_quarantined(self):
        """Only the top level is scanned, so re-running cannot re-process moved files."""
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir = Path(tmp)
            (npz_dir / INVALID_SUBDIR).mkdir()
            (npz_dir / INVALID_SUBDIR / "sg-one-north_aaa.npz").touch()
            self.assertEqual(index_npz_by_token(npz_dir), {})


class TestMoveInvalidNpz(unittest.TestCase):
    def _setup(self, tmp, rows, npz_tokens):
        npz_dir = Path(tmp)
        for token in npz_tokens:
            (npz_dir / f"sg-one-north_{token}.npz").touch()
        dataset = npz_dir / "routes.jsonl"
        _write_dataset(dataset, rows)
        return npz_dir, dataset

    def test_only_invalid_scenarios_are_moved(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(
                tmp, [("aaa", True), ("bbb", False), ("ccc", False)], ["aaa", "bbb", "ccc"]
            )
            summary = prune_npz(npz_dir, dataset)

            self.assertEqual(summary.moved_invalid, 2)
            self.assertTrue((npz_dir / "sg-one-north_aaa.npz").exists())
            self.assertFalse((npz_dir / "sg-one-north_bbb.npz").exists())
            quarantined = sorted(p.name for p in (npz_dir / INVALID_SUBDIR).iterdir())
            self.assertEqual(quarantined, ["sg-one-north_bbb.npz", "sg-one-north_ccc.npz"])

    def test_dry_run_touches_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, [("bbb", False)], ["bbb"])
            summary = prune_npz(npz_dir, dataset, dry_run=True)

            self.assertEqual(summary.moved_invalid, 1)
            self.assertTrue((npz_dir / "sg-one-north_bbb.npz").exists())
            self.assertFalse((npz_dir / INVALID_SUBDIR).exists())

    def test_invalid_scenarios_without_an_npz_are_counted_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, [("bbb", False), ("gone", False)], ["bbb"])
            summary = prune_npz(npz_dir, dataset)

            self.assertEqual(summary.moved_invalid, 1)
            self.assertEqual(summary.missing, 1)

    def test_rerunning_is_a_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, [("bbb", False)], ["bbb"])
            prune_npz(npz_dir, dataset)
            second = prune_npz(npz_dir, dataset)

            self.assertEqual(second.moved_invalid, 0)
            self.assertEqual(second.blocked, [])

    def test_an_existing_file_of_the_same_name_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, [("bbb", False)], ["bbb"])
            quarantine = npz_dir / INVALID_SUBDIR
            quarantine.mkdir()
            (quarantine / "sg-one-north_bbb.npz").write_text("earlier run")

            summary = prune_npz(npz_dir, dataset)

            self.assertEqual(summary.moved_invalid, 0)
            self.assertEqual(summary.blocked, ["sg-one-north_bbb.npz"])
            self.assertTrue((npz_dir / "sg-one-north_bbb.npz").exists())
            self.assertEqual((quarantine / "sg-one-north_bbb.npz").read_text(), "earlier run")

    def test_rows_without_an_explicit_flag_are_left_alone(self):
        """A row that never got a valid_route is unknown, not invalid."""
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir = Path(tmp)
            (npz_dir / "sg-one-north_ddd.npz").touch()
            dataset = npz_dir / "routes.jsonl"
            dataset.write_text(json.dumps({"token": "ddd", "routing_data": {}}) + "\n")

            summary = prune_npz(npz_dir, dataset)

            self.assertEqual(summary.moved_invalid, 0)
            self.assertTrue((npz_dir / "sg-one-north_ddd.npz").exists())


if __name__ == "__main__":
    unittest.main()


class TestNumScenariosCap(unittest.TestCase):
    def _setup(self, tmp, valid_tokens, invalid_tokens=()):
        npz_dir = Path(tmp)
        rows = [(t, True) for t in valid_tokens] + [(t, False) for t in invalid_tokens]
        for token in list(valid_tokens) + list(invalid_tokens):
            (npz_dir / f"sg-one-north_{token}.npz").touch()
        dataset = npz_dir / "routes.jsonl"
        _write_dataset(dataset, rows)
        return npz_dir, dataset

    def _top_level(self, npz_dir):
        return sorted(p.name for p in npz_dir.iterdir() if p.suffix == ".npz")

    def test_without_num_scenarios_every_valid_file_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a", "b", "c"])
            summary = prune_npz(npz_dir, dataset)

            self.assertEqual(summary.moved_surplus, 0)
            self.assertEqual(summary.kept, 3)
            self.assertFalse((npz_dir / SURPLUS_SUBDIR).exists())

    def test_excess_over_the_cap_is_moved_to_surplus(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a", "b", "c", "d", "e"])
            summary = prune_npz(npz_dir, dataset, num_scenarios=2)

            self.assertEqual(summary.moved_surplus, 3)
            self.assertEqual(summary.kept, 2)
            self.assertEqual(len(self._top_level(npz_dir)), 2)
            self.assertEqual(len(list((npz_dir / SURPLUS_SUBDIR).iterdir())), 3)

    def test_a_cap_above_the_available_count_moves_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a", "b"])
            summary = prune_npz(npz_dir, dataset, num_scenarios=10)

            self.assertEqual(summary.moved_surplus, 0)
            self.assertEqual(summary.kept, 2)

    def test_the_cap_counts_only_what_survives_the_invalid_pass(self):
        """Invalid files must not consume the budget."""
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a", "b", "c"], invalid_tokens=["x", "y"])
            summary = prune_npz(npz_dir, dataset, num_scenarios=2)

            self.assertEqual(summary.moved_invalid, 2)
            self.assertEqual(summary.moved_surplus, 1)
            self.assertEqual(summary.kept, 2)

    def test_the_same_seed_selects_the_same_files(self):
        selections = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                npz_dir, dataset = self._setup(tmp, [f"t{i}" for i in range(10)])
                prune_npz(npz_dir, dataset, num_scenarios=3, seed=7)
                selections.append(self._top_level(npz_dir))
        self.assertEqual(selections[0], selections[1])

    def test_a_different_seed_selects_differently(self):
        kept = []
        for seed in (1, 2):
            with tempfile.TemporaryDirectory() as tmp:
                npz_dir, dataset = self._setup(tmp, [f"t{i}" for i in range(20)])
                prune_npz(npz_dir, dataset, num_scenarios=5, seed=seed)
                kept.append(self._top_level(npz_dir))
        self.assertNotEqual(kept[0], kept[1])

    def test_dry_run_reports_the_cap_without_moving(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a", "b", "c", "d"], invalid_tokens=["x"])
            summary = prune_npz(npz_dir, dataset, num_scenarios=2, dry_run=True)

            self.assertEqual(summary.moved_invalid, 1)
            self.assertEqual(summary.moved_surplus, 2)
            self.assertEqual(summary.kept, 2)
            self.assertEqual(len(self._top_level(npz_dir)), 5)
            self.assertFalse((npz_dir / SURPLUS_SUBDIR).exists())
            self.assertFalse((npz_dir / INVALID_SUBDIR).exists())

    def test_capping_twice_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a", "b", "c", "d"])
            prune_npz(npz_dir, dataset, num_scenarios=2)
            first = self._top_level(npz_dir)
            second_summary = prune_npz(npz_dir, dataset, num_scenarios=2)

            self.assertEqual(second_summary.moved_surplus, 0)
            self.assertEqual(self._top_level(npz_dir), first)


class TestScenarioListRewrite(unittest.TestCase):
    """Training loads the scenario list, not the directory, so it has to follow the moves."""

    LIST_NAME = "scenarios_training.json"

    def _setup(self, tmp, valid_tokens, invalid_tokens=(), extra_entries=()):
        npz_dir = Path(tmp)
        rows = [(t, True) for t in valid_tokens] + [(t, False) for t in invalid_tokens]
        names = []
        for token in list(valid_tokens) + list(invalid_tokens):
            name = f"sg-one-north_{token}.npz"
            (npz_dir / name).touch()
            names.append(name)
        dataset = npz_dir / "routes.jsonl"
        _write_dataset(dataset, rows)
        (npz_dir / self.LIST_NAME).write_text(
            json.dumps(names + list(extra_entries)), encoding="utf-8"
        )
        return npz_dir, dataset

    def _entries(self, npz_dir, name=None):
        return json.loads((npz_dir / (name or self.LIST_NAME)).read_text(encoding="utf-8"))

    def test_invalid_scenarios_are_dropped_from_the_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a", "b"], ["bad"])
            prune_npz(npz_dir, dataset)
            self.assertEqual(self._entries(npz_dir), ["sg-one-north_a.npz", "sg-one-north_b.npz"])

    def test_surplus_scenarios_are_dropped_from_the_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a", "b", "c", "d"])
            prune_npz(npz_dir, dataset, num_scenarios=2)
            entries = self._entries(npz_dir)
            self.assertEqual(len(entries), 2)
            # Exactly the files still at the top level.
            self.assertEqual(
                sorted(entries), sorted(p.name for p in npz_dir.iterdir() if p.suffix == ".npz")
            )

    def test_the_original_is_kept_as_a_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a"], ["bad"])
            prune_npz(npz_dir, dataset)
            backup = npz_dir / (self.LIST_NAME + BACKUP_SUFFIX)
            self.assertTrue(backup.exists())
            self.assertEqual(len(self._entries(npz_dir, backup.name)), 2)

    def test_a_second_run_does_not_overwrite_the_backup_with_a_pruned_list(self):
        """The .bak must stay the pre-prune original, or the moves become unrecoverable."""
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a", "b", "c", "d"])
            prune_npz(npz_dir, dataset, num_scenarios=3)
            prune_npz(npz_dir, dataset, num_scenarios=2)
            backup = npz_dir / (self.LIST_NAME + BACKUP_SUFFIX)
            self.assertEqual(len(self._entries(npz_dir, backup.name)), 4)
            self.assertEqual(len(self._entries(npz_dir)), 2)

    def test_unrecognised_entries_survive(self):
        """Entries are subtracted, not rebuilt from the directory listing."""
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a"], ["bad"], extra_entries=["odd_name.npz"])
            prune_npz(npz_dir, dataset)
            self.assertIn("odd_name.npz", self._entries(npz_dir))

    def test_dry_run_leaves_the_list_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a"], ["bad"])
            summary = prune_npz(npz_dir, dataset, dry_run=True)
            self.assertEqual(len(self._entries(npz_dir)), 2)
            self.assertFalse((npz_dir / (self.LIST_NAME + BACKUP_SUFFIX)).exists())
            self.assertEqual(summary.scenario_lists[0].after, 1)

    def test_opting_out_leaves_the_list_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a"], ["bad"])
            summary = prune_npz(npz_dir, dataset, update_lists=False)
            self.assertEqual(len(self._entries(npz_dir)), 2)
            self.assertEqual(summary.scenario_lists, [])

    def test_a_backup_is_never_mistaken_for_a_list_to_prune(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a"], ["bad"])
            prune_npz(npz_dir, dataset)
            self.assertEqual([p.name for p in find_scenario_lists(npz_dir)], [self.LIST_NAME])

    def test_a_directory_without_a_list_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz_dir, dataset = self._setup(tmp, ["a"], ["bad"])
            (npz_dir / self.LIST_NAME).unlink()
            summary = prune_npz(npz_dir, dataset)
            self.assertEqual(summary.scenario_lists, [])
            self.assertEqual(summary.moved_invalid, 1)
