"""Describing a route the ego did not drive.

``build_routing_task_from_scenario`` reads the route off the driven trajectory, so it can only
describe what happened. These tests cover the path that takes the route as input instead, which is
what makes counterfactual-route datasets (alternative routes, re-routings) possible.
"""

import unittest
from unittest import mock

import route_description_generation.route_extraction as nu


class _Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    @property
    def point(self):
        return self


class _EgoState:
    def __init__(self, x=0.0, y=0.0, heading=0.0):
        self.rear_axle = _Pose(x, y, heading)


class _Pose:
    def __init__(self, x, y, heading):
        self.x, self.y, self.heading = x, y, heading

    @property
    def point(self):
        return _Point(self.x, self.y)


class _Scenario:
    token = "abc123"
    scenario_type = "on_carpark"
    _map_name = "us-ma-boston"

    def __init__(self):
        self.map_api = mock.Mock()
        self.initial_ego_state = _EgoState()


class TestBuildRoutingTaskFromAlternativeRoute(unittest.TestCase):
    ROUTE = ["rb1", "rb9"]
    REFERENCE = [(float(x), 0.0) for x in range(0, 101, 10)]

    def _task(self, **kwargs):
        with (
            mock.patch.object(nu, "project_goal_to_route_centerline", return_value=([80.0, 0.0], 0.0, "rb9")),
            mock.patch.object(nu, "project_start_to_route_midpoint", return_value=([0.0, 0.0], 0.0, "rb1")),
            mock.patch.object(nu, "build_route_centerline", return_value=self.REFERENCE),
            mock.patch.object(nu, "build_route_vias", return_value=None),
            mock.patch.object(nu, "route_connectivity_gaps", return_value=[]),
        ):
            return nu.build_routing_task_from_alternative_route(
                _Scenario(), self.ROUTE, (80.0, 0.0, 0.0), **kwargs
            )

    def test_task_describes_the_given_route_not_the_driven_one(self):
        task = self._task()
        self.assertEqual(task["route_roadblock_ids"], self.ROUTE)

    def test_reference_is_the_route_centerline_trimmed_to_the_ego_goal_stretch(self):
        """There is no driven path for a route nobody drove, so the shape check must differ.

        The centerline runs the whole route; only the ego-to-goal stretch is the reference, which
        is what keeps nuplan_route_length_m comparable with the OSRM leg.
        """
        task = self._task()
        self.assertEqual(task["reference_kind"], "centerline")
        self.assertEqual(task["route_polyline"], [(float(x), 0.0) for x in range(0, 81, 10)])
        self.assertAlmostEqual(task["nuplan_route_length_m"], 80.0)

        from route_description_generation.routing import PATH_CHECKS

        self.assertIn("centerline", PATH_CHECKS)

    def test_task_carries_the_fields_process_routing_task_reads(self):
        """build_routing_task_from_route omits these; its original caller never wrote a row."""
        task = self._task()
        for field in ("token", "scenario_type", "routing_horizon_s", "route_connectivity_gaps"):
            self.assertIn(field, task, field)
        self.assertEqual(task["routing_horizon_s"], 60.0)

    def test_token_defaults_to_the_scenario_but_can_be_overridden(self):
        """Several alternatives of one scenario share a scenario token and would overwrite in the
        index; each needs its own row token."""
        self.assertEqual(self._task()["token"], "abc123")
        self.assertEqual(self._task(token="abc123-alt0")["token"], "abc123-alt0")

    def test_goal_equals_goal_proj_so_the_request_ladder_drops_the_goal_varying_rung(self):
        task = self._task()
        self.assertEqual(task["goal"], task["goal_proj"])


class TestRouteDatasetWiring(unittest.TestCase):
    def test_the_command_is_registered(self):
        from route_description_generation.cli.main import cli

        self.assertIn("build-dataset-from-routes", cli.commands)

    def test_existing_commands_are_untouched(self):
        from route_description_generation.cli.main import cli

        for name in ("build-dataset", "export-tokens", "prune-npz", "scenario-filter", "modify-dataset"):
            self.assertIn(name, cli.commands)

    def test_route_rows_must_carry_what_the_builder_needs(self):
        import json
        import tempfile
        from pathlib import Path

        import click

        from route_description_generation.cli.build_dataset_from_routes import read_route_rows

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "routes.jsonl"
            path.write_text(json.dumps({"source_token": "a", "route_roadblock_ids": ["1"]}) + "\n")
            with self.assertRaises(click.ClickException):
                read_route_rows(path)


if __name__ == "__main__":
    unittest.main()
