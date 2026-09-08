import unittest
from unittest import mock

import requests

from route_description_generation import routing as rfn


class TestRouting(unittest.TestCase):
    def test_routing_selects_expected_osrm_port(self):
        cases = [
            ("us-ma-boston", "http://localhost:5001"),
            ("us-pa-pittsburgh-hazelwood", "http://localhost:5001"),
            ("us-nv-las-vegas-strip", "http://localhost:5002"),
            ("sg-one-north", "http://localhost:5003"),
        ]

        for map_name, expected_url in cases:
            with self.subTest(map_name=map_name):
                with mock.patch.object(rfn, "request_route", return_value={"code": "Ok"}) as m_osrm:
                    out = rfn.routing(
                        start=(1.0, 2.0),
                        destination=(3.0, 4.0),
                        map_name=map_name,
                        cs=None,
                        routing_service="osrm",
                    )
                    self.assertEqual(out, {"code": "Ok"})
                    self.assertEqual(m_osrm.call_args[0][2], expected_url)

    def test_routing_raises_for_unknown_map_name(self):
        with self.assertRaisesRegex(ValueError, "Unknown map_name"):
            rfn.routing(
                start=(1.0, 2.0),
                destination=(3.0, 4.0),
                map_name="unknown-map",
                routing_service="osrm",
            )

    def test_request_route_with_retry_retries_then_succeeds(self):
        side_effects = [
            requests.exceptions.ConnectionError("temporary"),
            requests.exceptions.ConnectionError("temporary"),
            {"ok": True},
        ]
        with mock.patch.object(rfn, "routing", side_effect=side_effects) as m_routing:
            with mock.patch.object(rfn.time, "sleep") as m_sleep:
                out = rfn.request_route_with_retry(foo="bar")

        self.assertEqual(out, {"ok": True})
        self.assertEqual(m_routing.call_count, 3)
        self.assertEqual(m_sleep.call_count, 2)

    def _request_route_task(self):
        return {
            "token": "tok123",
            "scenario_type": "unknown",
            "map_name": "sg-one-north",
            "start": (0.0, 1.0),
            "start_proj": (1.0, 2.0),
            "goal": (9.0, 19.0),
            "goal_proj": (10.0, 20.0),
            "epsg": 32648,
            "routing_horizon_s": 60.0,
            "start_heading": 0.25,
            "nuplan_route_length_m": 100.0,
            "route_vias": [(3.0, 4.0)],
        }

    def _run_ladder_with_deviations(self, distances, deviations):
        """Drive the ladder with scripted OSRM distances and ``(p95, mean)`` path deviations."""
        task = self._request_route_task()
        task["route_polyline"] = [(0.0, 0.0), (100.0, 0.0)]
        remaining_distances = list(distances)
        remaining_deviations = list(deviations)

        def _fake_execute(start_abs, goal_abs, map_name, epsg, **kwargs):
            return (
                {"routes": [{"distance": remaining_distances.pop(0)}]},
                "desc",
                [[0.0, 0.0]],
            )

        def _fake_deviation(routing_output, epsg, route_polyline):
            p95, mean = remaining_deviations.pop(0)
            return {
                "path_deviation_max_m": p95,
                "path_deviation_p95_m": p95,
                "path_deviation_mean_m": mean,
            }

        with (
            mock.patch.object(rfn, "execute_routing_request", side_effect=_fake_execute) as m_exec,
            mock.patch.object(rfn, "osrm_path_deviation", side_effect=_fake_deviation),
        ):
            result = rfn.process_routing_task(task, routing_service="osrm")
        return result, m_exec

    def test_close_matching_attempt_short_circuits_the_ladder(self):
        """A rung agreeing on both length and shape is accepted without trying the rest."""
        result, m_exec = self._run_ladder_with_deviations(
            distances=[100.0] * 6, deviations=[(3.0, 1.5)] * 6
        )
        self.assertEqual(m_exec.call_count, 1)
        self.assertTrue(result["routing_data"]["valid_route"])
        self.assertEqual(result["routing_data"]["routing_strategy"], "A_vias_start_proj")
        self.assertEqual(result["routing_data"]["route_validation"]["path_deviation_p95_m"], 3.0)

    def test_localised_excursion_does_not_shadow_a_closer_later_attempt(self):
        """A rung whose mean is fine but p95 is not must not hide a rung that hugs the path.

        This is the u-turn-at-the-wrong-median case: the route is right on average but takes one
        wide excursion, and a later rung follows the driven path throughout.
        """
        result, m_exec = self._run_ladder_with_deviations(
            distances=[100.0] * 6,
            deviations=[(44.0, 3.3), (44.0, 3.3), (2.6, 1.8), (2.6, 1.8), (44.0, 3.3), (5.0, 2.0)],
        )
        # A and B pass both bars but exceed the accept threshold, so the walk continues to C.
        self.assertEqual(m_exec.call_count, 3)
        self.assertTrue(result["routing_data"]["valid_route"])
        self.assertEqual(result["routing_data"]["routing_strategy"], "C_no_vias_start_proj")
        self.assertEqual(result["routing_data"]["route_validation"]["path_deviation_p95_m"], 2.6)

    def test_closest_attempt_wins_when_none_clears_the_threshold(self):
        """With no rung below the accept threshold, the one nearest the driven path wins."""
        result, m_exec = self._run_ladder_with_deviations(
            distances=[100.0] * 6,
            deviations=[
                (60.0, 5.0),
                (55.0, 5.0),
                (30.0, 5.0),
                (70.0, 5.0),
                (80.0, 5.0),
                (65.0, 5.0),
            ],
        )
        self.assertEqual(m_exec.call_count, 6)
        self.assertEqual(result["routing_data"]["routing_strategy"], "C_no_vias_start_proj")
        self.assertEqual(result["routing_data"]["route_validation"]["path_deviation_p95_m"], 30.0)

    def test_length_invalid_attempts_are_never_selected_on_geometry(self):
        """Geometry ranks length-agreeing rungs; it does not rescue one that failed on length."""
        # A-C hug the driven path but are far too long; D agrees on length and on shape.
        result, _ = self._run_ladder_with_deviations(
            distances=[10000.0, 10000.0, 10000.0, 100.0, 100.0, 100.0],
            deviations=[(1.0, 0.5)] * 6,
        )
        self.assertEqual(result["routing_data"]["routing_strategy"], "D_no_vias_start")
        self.assertTrue(result["routing_data"]["valid_route"])

    def test_route_wandering_off_the_driven_path_is_invalid_despite_matching_length(self):
        """The Capstan Way case: every rung agrees on length while describing another street."""
        result, m_exec = self._run_ladder_with_deviations(
            distances=[100.0] * 6, deviations=[(70.0, 21.6)] * 6
        )
        self.assertEqual(m_exec.call_count, 6)
        self.assertFalse(result["routing_data"]["valid_route"])
        validation = result["routing_data"]["route_validation"]
        self.assertTrue(validation["length_valid"])
        self.assertFalse(validation["path_valid"])

    def test_raw_start_and_goal_rung_uses_the_unprojected_pair(self):
        """F must send the raw goal, and record it as the route end when it wins."""
        task = self._request_route_task()
        task["route_polyline"] = [(0.0, 0.0), (100.0, 0.0)]
        # Only the final rung agrees on length.
        distances = [10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 100.0]

        def _fake_execute(start_abs, goal_abs, map_name, epsg, **kwargs):
            return ({"routes": [{"distance": distances.pop(0)}]}, "desc", [[0.0, 0.0]])

        with mock.patch.object(rfn, "execute_routing_request", side_effect=_fake_execute) as m_exec:
            result = rfn.process_routing_task(task, routing_service="osrm")

        self.assertEqual(m_exec.call_count, 6)
        starts_and_goals = [(c.args[0], c.args[1]) for c in m_exec.call_args_list]
        self.assertEqual(starts_and_goals[-1], (task["start"], task["goal"]))
        # Every earlier rung routes to the projected goal.
        for _start, goal in starts_and_goals[:-1]:
            self.assertEqual(goal, task["goal_proj"])
        self.assertEqual(result["routing_data"]["routing_strategy"], "F_no_vias_raw_start_and_goal")
        self.assertEqual(result["routing_data"]["route_end"], list(task["goal"]))

    def test_unsupported_routing_service_is_rejected(self):
        """OSRM is the only backend; anything else must fail loudly rather than silently."""
        with self.assertRaises(ValueError):
            rfn.process_routing_task(self._request_route_task(), routing_service="gmaps")
        with self.assertRaises(ValueError):
            rfn.routing_to_language({}, routing_service="gmaps")

    def test_process_routing_task_escalates_through_all_attempts(self):
        """Attempts escalate vias->no-vias, projected start->raw ego start, then wide bearing."""
        task = self._request_route_task()

        # Always far too long vs. nuplan_route_length_m=100 -> every attempt fails validation,
        # so the full escalation ladder runs.
        def _fake_execute(start_abs, goal_abs, map_name, epsg, **kwargs):
            return ({"routes": [{"distance": 10000.0}]}, "desc", [[0.0, 0.0]])

        with mock.patch.object(rfn, "execute_routing_request", side_effect=_fake_execute) as m_exec:
            result = rfn.process_routing_task(task, routing_service="osrm")

        self.assertEqual(m_exec.call_count, 6)
        attempts = [
            (
                call.args[0],
                call.args[1],
                call.kwargs.get("vias"),
                call.kwargs.get("bearing_tol"),
            )
            for call in m_exec.call_args_list
        ]
        narrow = rfn.BEARING_TOL_DEG
        proj_goal = task["goal_proj"]
        self.assertEqual(
            attempts,
            [
                ((1.0, 2.0), proj_goal, task["route_vias"], narrow),  # A: proj start, vias
                ((0.0, 1.0), proj_goal, task["route_vias"], narrow),  # B: raw start, vias
                ((1.0, 2.0), proj_goal, None, narrow),  # C: proj start, no vias
                ((0.0, 1.0), proj_goal, None, narrow),  # D: raw start, no vias
                # E: back to A's configuration, but with the widened bearing cone.
                ((1.0, 2.0), proj_goal, task["route_vias"], rfn.BEARING_TOL_WIDE_DEG),
                # F: raw start AND raw goal - the only self-consistent unprojected pair.
                ((0.0, 1.0), task["goal"], None, narrow),
            ],
        )
        self.assertFalse(result["routing_data"]["valid_route"])
        # When nothing validates, the attempt closest to the driven path is reported. Here no
        # geometry is available, so every attempt ties and the last one wins.
        self.assertEqual(result["routing_data"]["routing_strategy"], "F_no_vias_raw_start_and_goal")
        # The kept attempt's start is what gets recorded as the route start.
        self.assertEqual(result["routing_data"]["route_start"], [0.0, 1.0])

    def test_process_routing_task_keeps_wide_bearing_attempt_when_it_validates(self):
        """E is kept only when it passes: it rescues a run that A-D all failed."""
        task = self._request_route_task()
        # A-D far too long vs. nuplan_route_length_m=100; E agrees -> E is the kept result.
        distances = [10000.0, 10000.0, 10000.0, 10000.0, 95.0]

        def _fake_execute(start_abs, goal_abs, map_name, epsg, **kwargs):
            return ({"routes": [{"distance": distances.pop(0)}]}, "desc", [[0.0, 0.0]])

        with mock.patch.object(rfn, "execute_routing_request", side_effect=_fake_execute) as m_exec:
            result = rfn.process_routing_task(task, routing_service="osrm")

        self.assertEqual(m_exec.call_count, 5)
        self.assertTrue(result["routing_data"]["valid_route"])
        self.assertEqual(
            result["routing_data"]["routing_strategy"], "E_vias_start_proj_wide_bearing"
        )
        self.assertEqual(result["routing_data"]["route_start"], [1.0, 2.0])

    def test_process_routing_task_stops_at_first_valid_attempt(self):
        """A failing first attempt falls through, and the walk stops as soon as one validates."""
        task = self._request_route_task()
        # 1st attempt way off, 2nd attempt agrees with nuplan_route_length_m -> stop at B.
        distances = [10000.0, 95.0]

        def _fake_execute(start_abs, goal_abs, map_name, epsg, **kwargs):
            return (
                {"routes": [{"distance": distances.pop(0)}]},
                "desc",
                [[0.0, 0.0]],
            )

        with mock.patch.object(rfn, "execute_routing_request", side_effect=_fake_execute) as m_exec:
            result = rfn.process_routing_task(task, routing_service="osrm")

        self.assertEqual(m_exec.call_count, 2)
        self.assertTrue(result["routing_data"]["valid_route"])
        self.assertEqual(result["routing_data"]["routing_strategy"], "B_vias_start")
        self.assertEqual(result["routing_data"]["route_start"], [0.0, 1.0])

    def test_process_routing_task_adds_interplan_payload(self):
        routing_task = {
            "token": "tok123",
            "scenario_type": "unknown",
            "map_name": "sg-one-north",
            "start": (0.0, 1.0),
            "start_proj": (1.0, 2.0),
            "goal": (9.0, 19.0),
            "goal_proj": (10.0, 20.0),
            "epsg": 32648,
            "routing_horizon_s": 60.0,
            "start_heading": 0.25,
            "nuplan_route_length_m": 100.0,
            "route_vias": [(3.0, 4.0)],
            "interplan_goals": {
                "left": (11.0, 21.0),
                "right": None,
                "straight": (12.0, 22.0),
            },
        }

        def _fake_execute(start_abs, goal_abs, map_name, epsg, **kwargs):
            return (
                {"route_to": list(goal_abs), "routes": [{"distance": 95.0}]},
                f"desc-{goal_abs[0]:.0f}",
                [[goal_abs[0], goal_abs[1]]],
            )

        with mock.patch.object(rfn, "execute_routing_request", side_effect=_fake_execute) as m_exec:
            result = rfn.process_routing_task(routing_task, routing_service="osrm")

        self.assertIn("routing_data", result)
        self.assertIn("routing_data_interplan", result)
        self.assertEqual(result["routing_data"]["route_end"], [10.0, 20.0])

        interplan_payload = result["routing_data_interplan"]
        self.assertEqual(interplan_payload["route_end_left"], [11.0, 21.0])
        self.assertIsNone(interplan_payload["route_end_right"])
        self.assertEqual(interplan_payload["route_end_straight"], [12.0, 22.0])
        self.assertEqual(interplan_payload["route_description_left"], "desc-11")
        self.assertIsNone(interplan_payload["route_description_right"])
        self.assertTrue(result["routing_data"]["valid_route"])
        self.assertEqual(result["routing_data"]["routing_strategy"], "A_vias_start_proj")
        # The first attempt validates, so the projected start is what gets recorded.
        self.assertEqual(result["routing_data"]["route_start"], [1.0, 2.0])

        # Main + left + straight calls.
        self.assertEqual(m_exec.call_count, 3)
        # A-phase main request forwards the task's route_vias.
        self.assertEqual(m_exec.call_args_list[0].kwargs.get("vias"), routing_task["route_vias"])
        # Interplan variant requests (left/straight) are plain two-point requests.
        self.assertIsNone(m_exec.call_args_list[1].kwargs.get("vias"))
        self.assertIsNone(m_exec.call_args_list[2].kwargs.get("vias"))


class TestSelectRouteWithoutAReference(unittest.TestCase):
    """A caller with no ground truth must not pay for a ladder it cannot judge."""

    def _select(self, task_overrides):
        task = {
            "map_name": "sg-one-north",
            "start": (0.0, 1.0),
            "start_proj": (1.0, 2.0),
            "goal": (9.0, 19.0),
            "goal_proj": (10.0, 20.0),
            "epsg": 32648,
            "start_heading": 0.25,
            "route_vias": [(3.0, 4.0)],
        }
        task.update(task_overrides)
        with mock.patch.object(
            rfn,
            "execute_routing_request",
            return_value=({"routes": [{"distance": 500.0}]}, "desc", [[0.0, 0.0]]),
        ) as m_exec:
            selection = rfn.select_route(task)
        return selection, m_exec

    def test_no_length_and_no_path_issues_exactly_one_request(self):
        selection, m_exec = self._select({"nuplan_route_length_m": None, "route_polyline": []})
        self.assertEqual(m_exec.call_count, 1)
        self.assertEqual(selection.strategy, "A_vias_start_proj")

    def test_the_single_request_is_the_strongest_configuration(self):
        """Falling through to the last rung would silently change what an unvalidated caller gets."""
        selection, _ = self._select({"nuplan_route_length_m": None, "route_polyline": None})
        self.assertEqual(selection.start, (1.0, 2.0))
        self.assertEqual(selection.goal, (10.0, 20.0))
        self.assertEqual(selection.description, "desc")

    def test_a_length_alone_is_enough_to_walk_the_ladder(self):
        _, m_exec = self._select({"nuplan_route_length_m": 100.0, "route_polyline": []})
        self.assertGreater(m_exec.call_count, 1)

    def test_a_path_alone_is_enough_to_walk_the_ladder(self):
        _, m_exec = self._select(
            {"nuplan_route_length_m": None, "route_polyline": [(0.0, 0.0), (100.0, 0.0)]}
        )
        self.assertGreater(m_exec.call_count, 1)


class TestPathCheckSelection(unittest.TestCase):
    """Which deviation statistic gates a route depends on what the reference was derived from."""

    def _validate(self, deviations, reference_kind=None):
        task = {
            "map_name": "sg-one-north",
            "start": (0.0, 1.0),
            "start_proj": (1.0, 2.0),
            "goal": (10.0, 20.0),
            "goal_proj": (10.0, 20.0),
            "epsg": 32648,
            "start_heading": 0.25,
            "nuplan_route_length_m": 100.0,
            "route_polyline": [(0.0, 0.0), (100.0, 0.0)],
            "route_vias": None,
        }
        if reference_kind is not None:
            task["reference_kind"] = reference_kind
        with (
            mock.patch.object(
                rfn,
                "execute_routing_request",
                return_value=({"routes": [{"distance": 100.0}]}, "desc", []),
            ),
            mock.patch.object(rfn, "osrm_path_deviation", return_value=deviations),
        ):
            return rfn.select_route(task)

    def _dev(self, max_m, p95, mean):
        return {
            "path_deviation_max_m": max_m,
            "path_deviation_p95_m": p95,
            "path_deviation_mean_m": mean,
        }

    def test_a_trajectory_reference_is_gated_on_the_mean(self):
        # p95 far above the centerline limit, mean below the trajectory limit -> accepted.
        sel = self._validate(self._dev(60.0, 45.0, 5.0), reference_kind="trajectory")
        self.assertTrue(sel.validation["path_valid"])

    def test_a_centerline_reference_is_gated_on_the_p95(self):
        # The same numbers under a centerline reference: the p95 exceeds its limit -> rejected.
        sel = self._validate(self._dev(60.0, 45.0, 5.0), reference_kind="centerline")
        self.assertFalse(sel.validation["path_valid"])

    def test_a_centerline_tolerates_the_offset_that_would_fail_the_mean_gate(self):
        """A centerline's lane-width bias lands in the mean, which is why it is not the gate here."""
        inflated_mean = self._dev(25.0, 18.0, 14.0)
        self.assertTrue(self._validate(inflated_mean, "centerline").validation["path_valid"])
        self.assertFalse(self._validate(inflated_mean, "trajectory").validation["path_valid"])

    def test_the_default_is_the_trajectory_check(self):
        """rdg's own tasks carry no reference_kind and must keep the mean gate."""
        omitted = self._validate(self._dev(25.0, 18.0, 14.0))
        explicit = self._validate(self._dev(25.0, 18.0, 14.0), reference_kind="trajectory")
        self.assertEqual(omitted.validation["path_valid"], explicit.validation["path_valid"])
        self.assertFalse(omitted.validation["path_valid"])


class TestRouteAttempts(unittest.TestCase):
    LADDER = dict(start=(0.0, 0.0), start_proj=(1.0, 1.0), vias=[(5.0, 5.0, 0.0)])

    def test_a_projected_goal_gets_the_full_six_rung_ladder(self):
        attempts = rfn.route_attempts(goal=(9.0, 9.0), goal_proj=(8.0, 8.0), **self.LADDER)
        self.assertEqual(
            [a.strategy for a in attempts],
            [
                "A_vias_start_proj",
                "B_vias_start",
                "C_no_vias_start_proj",
                "D_no_vias_start",
                "E_vias_start_proj_wide_bearing",
                "F_no_vias_raw_start_and_goal",
            ],
        )

    def test_a_fixed_goal_drops_the_rung_that_would_duplicate_d(self):
        """With goal == goal_proj, F sends exactly D's request, so it is a wasted round trip."""
        attempts = rfn.route_attempts(goal=(8.0, 8.0), goal_proj=(8.0, 8.0), **self.LADDER)
        self.assertEqual(len(attempts), 5)
        self.assertNotIn("F_no_vias_raw_start_and_goal", [a.strategy for a in attempts])

        d = next(a for a in attempts if a.strategy == "D_no_vias_start")
        would_be_f = (self.LADDER["start"], (8.0, 8.0), None, rfn.BEARING_TOL_DEG)
        self.assertEqual((d.start, d.goal, d.vias, d.bearing_tol), would_be_f)

    def test_only_e_widens_the_bearing_cone(self):
        attempts = rfn.route_attempts(goal=(9.0, 9.0), goal_proj=(8.0, 8.0), **self.LADDER)
        widened = [a.strategy for a in attempts if a.bearing_tol == rfn.BEARING_TOL_WIDE_DEG]
        self.assertEqual(widened, ["E_vias_start_proj_wide_bearing"])

    def test_e_repeats_a_apart_from_the_cone(self):
        attempts = {
            a.strategy: a
            for a in rfn.route_attempts(goal=(9.0, 9.0), goal_proj=(8.0, 8.0), **self.LADDER)
        }
        a, e = attempts["A_vias_start_proj"], attempts["E_vias_start_proj_wide_bearing"]
        self.assertEqual((a.start, a.goal, a.vias), (e.start, e.goal, e.vias))
        self.assertNotEqual(a.bearing_tol, e.bearing_tol)


if __name__ == "__main__":
    unittest.main()
