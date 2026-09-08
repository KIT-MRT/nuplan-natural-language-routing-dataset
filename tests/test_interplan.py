import tempfile
import unittest
from pathlib import Path

from route_description_generation.interplan import (
    load_interplan_modification_goals,
    load_interplan_tokens,
    resolve_interplan_yaml_paths,
)


class TestInterplanIntegration(unittest.TestCase):
    def test_load_interplan_modification_goals(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mod_yaml = Path(tmp_dir) / "interPlan_modifications.yaml"
            mod_yaml.write_text(
                """
modification_details_dictionary:
  tok_a:
    goal:
      left: "1.0, 2.0"
      right: null
      straight: "3.5,4.5"
""".strip(),
                encoding="utf-8",
            )

            goals = load_interplan_modification_goals(mod_yaml)
            self.assertEqual(goals["tok_a"]["left"], (1.0, 2.0))
            self.assertIsNone(goals["tok_a"]["right"])
            self.assertEqual(goals["tok_a"]["straight"], (3.5, 4.5))

    def test_load_interplan_tokens_unions_and_normalizes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_yaml = root / "benchmark_scenarios.yaml"
            modifications_yaml = root / "interPlan_modifications.yaml"

            benchmark_yaml.write_text(
                """
scenario_tokens:
  - abc123-s0
  - def456-lg
""".strip(),
                encoding="utf-8",
            )

            modifications_yaml.write_text(
                """
modification_details_dictionary:
  ghi789:
    goal:
      left: null
      right: null
      straight: null
""".strip(),
                encoding="utf-8",
            )

            tokens = load_interplan_tokens(benchmark_yaml, modifications_yaml)
            self.assertEqual(tokens, ["abc123", "def456", "ghi789"])

    def test_resolve_interplan_yaml_paths_no_interplan_import(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            benchmark_yaml = root / "benchmark_scenarios.yaml"
            modifications_yaml = root / "interPlan_modifications.yaml"
            benchmark_yaml.write_text("scenario_tokens: []", encoding="utf-8")
            modifications_yaml.write_text("modification_details_dictionary: {}", encoding="utf-8")

            resolved_benchmark, resolved_mod = resolve_interplan_yaml_paths(
                benchmark_yaml,
                modifications_yaml,
            )

            self.assertEqual(resolved_benchmark, benchmark_yaml)
            self.assertEqual(resolved_mod, modifications_yaml)


if __name__ == "__main__":
    unittest.main()
