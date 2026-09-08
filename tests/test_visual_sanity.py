import json
import os
import random
import textwrap
import unittest
from pathlib import Path

import pytest
from nuplan.common.actor_state.state_representation import Point2D
from tqdm import tqdm

DATASET = "val14_lg_routing_data.jsonl"


@pytest.mark.integration
class TestRoutingVisualSanity(unittest.TestCase):
    def test_val14_examples_visualize_and_have_rough_length_agreement(self):
        data_root = os.environ.get("NUPLAN_DATA_ROOT")
        maps_root = os.environ.get("NUPLAN_MAPS_ROOT")
        if not data_root or not maps_root:
            self.skipTest("Set NUPLAN_DATA_ROOT and NUPLAN_MAPS_ROOT to run visual sanity test")
        if not (os.path.isdir(data_root) and os.path.isdir(maps_root)):
            self.skipTest("NUPLAN_DATA_ROOT/NUPLAN_MAPS_ROOT do not point to valid directories")

        package_root = Path(__file__).resolve().parents[1]
        dataset_file = package_root / "datasets" / DATASET
        if not dataset_file.exists():
            self.skipTest(f"Missing dataset file: {dataset_file}")

        from nucontrol.scenario_query import ScenarioLoader
        from nucontrol.visualize import plot_route

        from route_description_generation.waypoints import (
            build_route_vias,
        )

        valid_sample_count_raw = os.environ.get("ROUTING_VISUAL_SANITY_VALID_SAMPLE_COUNT", "10")
        try:
            valid_sample_count = int(valid_sample_count_raw)
        except ValueError as err:
            raise AssertionError(
                "ROUTING_VISUAL_SANITY_VALID_SAMPLE_COUNT must be an integer"
            ) from err
        if valid_sample_count < 0:
            raise AssertionError("ROUTING_VISUAL_SANITY_VALID_SAMPLE_COUNT must be >= 0")

        # Caps how many invalid scenarios actually get loaded+plotted (the expensive part), so a
        # default/quick test run stays fast even when the dataset has many invalid rows. The
        # pass/fail check below still always considers every invalid row found, regardless of
        # this limit — only the number of PNGs generated is affected. "all"/"0" disables the cap.
        invalid_plot_limit_raw = os.environ.get("ROUTING_VISUAL_SANITY_INVALID_PLOT_LIMIT", "10")
        if invalid_plot_limit_raw.lower() in {"all", "0"}:
            invalid_plot_limit = None
        else:
            try:
                invalid_plot_limit = int(invalid_plot_limit_raw)
            except ValueError as err:
                raise AssertionError(
                    "ROUTING_VISUAL_SANITY_INVALID_PLOT_LIMIT must be an integer, 0, or 'all'"
                ) from err
            if invalid_plot_limit < 0:
                raise AssertionError("ROUTING_VISUAL_SANITY_INVALID_PLOT_LIMIT must be >= 0")

        # Classify every row by valid_route straight from the dataset (cheap JSON parsing, no
        # scenario loading). Scenario loading only happens below, for the two (usually much
        # smaller) sets we actually visualize: every invalid row, plus a random valid sample —
        # so runtime no longer scales with the size of the whole dataset file.
        invalid_examples = []
        valid_examples = []
        with dataset_file.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rd = row.get("routing_data") or {}
                directions = rd.get("route_directions") or {}
                routes = directions.get("routes") or []
                if not routes:
                    continue
                osrm_dist = routes[0].get("distance")
                if not isinstance(osrm_dist, (int, float)) or osrm_dist <= 0:
                    continue
                goal_abs = rd.get("route_end")
                if (
                    not isinstance(goal_abs, list)
                    or len(goal_abs) != 2
                    or not all(isinstance(v, (int, float)) for v in goal_abs)
                ):
                    continue
                route_validation = rd.get("route_validation") or {}
                nuplan_len = route_validation.get("nuplan_length_m")
                if not isinstance(nuplan_len, (int, float)):
                    continue
                route_roadblock_ids = rd.get("route_roadblock_ids")
                if not route_roadblock_ids:
                    continue
                example = {
                    "token": row["token"],
                    "description": rd.get("route_description", ""),
                    "osrm_distance": float(osrm_dist),
                    "goal_abs": [float(goal_abs[0]), float(goal_abs[1])],
                    "route_maneuver_positions": rd.get("route_maneuver_positions", []),
                    "nuplan_len": float(nuplan_len),
                    "route_validation": route_validation,
                    "route_roadblock_ids": route_roadblock_ids,
                }
                if rd.get("valid_route"):
                    valid_examples.append(example)
                else:
                    invalid_examples.append(example)

        if not invalid_examples and not valid_examples:
            self.skipTest(f"No usable examples found in {DATASET}")

        loader = ScenarioLoader(
            data_root=data_root,
            map_root=maps_root,
            include_splits=["trainval"],
            log_split="val",
            extraction_offset=0.0,
            max_workers=8,
        )

        out_dir = package_root / "tests" / "output" / "routing_visual_sanity"
        valid_dir = out_dir / "valid_random"
        failed_dir = out_dir / "failed"
        valid_dir.mkdir(parents=True, exist_ok=True)
        failed_dir.mkdir(parents=True, exist_ok=True)
        for old_png in valid_dir.glob("*.png"):
            old_png.unlink()
        for old_png in failed_dir.glob("*.png"):
            old_png.unlink()

        def plot_example(ex, valid: bool, out_path: Path) -> None:
            scenario = loader.load_scenario(ex["token"])
            goal_point = Point2D(ex["goal_abs"][0], ex["goal_abs"][1])
            route_vias = build_route_vias(
                scenario.map_api,
                ex["route_roadblock_ids"],
                scenario.initial_ego_state.rear_axle.point,
                goal_point=goal_point,
            )
            # Vias carry (x, y, heading); plot_route only takes positions.
            if route_vias:
                route_vias = [(v[0], v[1]) for v in route_vias]
            ratio = ex["route_validation"].get("ratio_nuplan_over_osrm")
            ratio_str = f"{ratio:.3f}" if ratio is not None else "n/a"
            goal = (ex["goal_abs"][0], ex["goal_abs"][1], 0.0)
            title = (
                f"{'PASS' if valid else 'FAIL'} {ex['token']}"
                f"\nOSRM={ex['osrm_distance']:.1f}m | nuPlan={ex['nuplan_len']:.1f}m | "
                f"ratio={ratio_str}"
                f"\n{textwrap.fill(ex['description'], width=120)}"
            )
            plot_route(
                scenario=scenario,
                roadblock_ids=ex["route_roadblock_ids"],
                goal=goal,
                title=title,
                out_path=str(out_path),
                vias=route_vias,
                maneuver_positions=ex.get("route_maneuver_positions") if not valid else None,
            )

        # Visualize invalid scenarios for inspection (up to invalid_plot_limit; None = all).
        invalid_to_plot = (
            invalid_examples
            if invalid_plot_limit is None
            else invalid_examples[:invalid_plot_limit]
        )
        for ex in tqdm(invalid_to_plot, desc="Plotting invalid routes", unit="scenario"):
            try:
                plot_example(
                    ex,
                    valid=False,
                    out_path=failed_dir / f"{ex['token']}_failed_route_lanes.png",
                )
            except Exception:
                # Best-effort diagnostics only.
                pass

        # Visualize a random sample of valid scenarios for a visual spot-check.
        sample_size = min(valid_sample_count, len(valid_examples))
        for ex in tqdm(
            random.sample(valid_examples, sample_size),
            desc="Plotting valid sample",
            unit="scenario",
        ):
            plot_example(
                ex,
                valid=True,
                out_path=valid_dir / f"{ex['token']}_valid_route_lanes.png",
            )

        self.assertTrue(
            valid_examples,
            msg=(
                f"No scenarios passed the visual sanity checks. See failure plots in: {failed_dir}"
            ),
        )
        first_failures = ", ".join(ex["token"] for ex in invalid_examples[:5])
        plot_note = (
            f"{len(invalid_to_plot)} plotted (ROUTING_VISUAL_SANITY_INVALID_PLOT_LIMIT="
            f"{invalid_plot_limit}). "
            if invalid_plot_limit is not None and len(invalid_examples) > len(invalid_to_plot)
            else ""
        )
        self.assertFalse(
            invalid_examples,
            msg=(
                f"{len(invalid_examples)} / {len(invalid_examples) + len(valid_examples)} "
                f"scenarios have valid_route=False. "
                f"{plot_note}"
                f"Failed plots: {failed_dir}. "
                f"Random valid plots: {valid_dir}. "
                f"First failures: {first_failures}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
