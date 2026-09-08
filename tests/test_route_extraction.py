import math
import unittest
from unittest import mock

from route_description_generation import route_extraction as nu


class _Point:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _RearAxle:
    def __init__(self, x, y):
        self.point = _Point(x, y)


class _State:
    """Minimal stand-in for an EgoState (only rear_axle.point is used)."""

    def __init__(self, x, y):
        self.rear_axle = _RearAxle(x, y)


class _Pose:
    def __init__(self, x, y, heading):
        self.x, self.y, self.heading = x, y, heading


class _Lane:
    """Lane whose baseline is a single pose per contained point, all at the same heading."""

    def __init__(self, points, heading):
        self.baseline_path = mock.Mock()
        self.baseline_path.discrete_path = [_Pose(x, y, heading) for x, y in points]


class _Roadblock:
    def __init__(self, rb_id, contains=(), outgoing=(), heading=None):
        self.id = rb_id
        self._contains = set(contains)
        self._outgoing_ids = list(outgoing)
        self.outgoing_edges = []
        # Only roadblocks used in heading-disambiguation tests need a geometry.
        self.interior_edges = [_Lane(self._contains, heading)] if heading is not None else []

    def contains_point(self, point):
        return (point.x, point.y) in self._contains


class _FakeMap:
    """Roadblock graph keyed by id, with `a -> b` edges resolved after construction."""

    def __init__(self, roadblocks):
        self.by_id = {rb.id: rb for rb in roadblocks}
        for rb in roadblocks:
            rb.outgoing_edges = [self.by_id[o] for o in rb._outgoing_ids if o in self.by_id]

    def get_map_object(self, rb_id, layer):
        # Every object resolves on the first layer queried; the second lookup is unnecessary.
        return self.by_id.get(rb_id)

    def get_proximal_map_objects(self, point, radius, layers):
        return {layers[0]: [rb for rb in self.by_id.values() if rb.contains_point(point)]}


class TestBuildRouteFromTrajectory(unittest.TestCase):
    def _build(self, fake_map, raw_ids):
        with mock.patch.object(nu, "get_roadblock_ids_from_trajectory", return_value=raw_ids):
            return nu.build_route_roadblock_ids_from_trajectory(fake_map, [])

    def test_already_connected_route_is_unchanged(self):
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["B"]),
                _Roadblock("B", outgoing=["C"]),
                _Roadblock("C"),
            ]
        )
        self.assertEqual(self._build(fake_map, ["A", "B", "C"]), ["A", "B", "C"])

    def test_skipped_roadblock_is_bridged(self):
        # The devkit reports A then C, skipping B in an ambiguous area.
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["B"]),
                _Roadblock("B", outgoing=["C"]),
                _Roadblock("C"),
            ]
        )
        self.assertEqual(self._build(fake_map, ["A", "C"]), ["A", "B", "C"])

    def test_multi_hop_gap_is_bridged(self):
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["B"]),
                _Roadblock("B", outgoing=["C"]),
                _Roadblock("C", outgoing=["D"]),
                _Roadblock("D"),
            ]
        )
        self.assertEqual(self._build(fake_map, ["A", "D"]), ["A", "B", "C", "D"])

    def test_unbridgeable_gap_keeps_both_ends(self):
        # Mirrors us-ma-boston 49253 -> 48808: no outgoing edge exists, yet the ego drove it.
        fake_map = _FakeMap([_Roadblock("A"), _Roadblock("B")])
        route = self._build(fake_map, ["A", "B"])
        self.assertEqual(route, ["A", "B"])
        self.assertEqual(nu.route_connectivity_gaps(fake_map, route), [["A", "B"]])

    def test_spurious_parallel_connector_is_skipped(self):
        # X is an overlapping junction connector the ego's points strayed onto; it is reachable
        # from nothing on the route, while the real continuation B is.
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["B"]),
                _Roadblock("B"),
                _Roadblock("X"),
            ]
        )
        self.assertEqual(self._build(fake_map, ["A", "X", "B"]), ["A", "B"])

    def test_run_of_parallel_connectors_is_skipped(self):
        # Mirrors 19458 -> 19460 -> 19461 -> 19457: a run of mutually unreachable alternatives.
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["B"]),
                _Roadblock("B"),
                _Roadblock("X"),
                _Roadblock("Y"),
                _Roadblock("Z"),
            ]
        )
        self.assertEqual(self._build(fake_map, ["A", "X", "Y", "Z", "B"]), ["A", "B"])

    def test_committed_stray_is_backtracked(self):
        # Mirror of the lookahead case: the stray X was already committed, so nothing after it
        # connects. Dropping it restores A -> B.
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["B"]),
                _Roadblock("B"),
                _Roadblock("X"),
            ]
        )
        self.assertEqual(self._build(fake_map, ["A", "X", "B"]), ["A", "B"])

    def test_leading_break_is_kept_not_dropped(self):
        # A stray at the very start is indistinguishable from a genuine break there, so the
        # prefix is never discarded — the break is reported instead of silently losing roadblocks.
        fake_map = _FakeMap(
            [
                _Roadblock("X"),
                _Roadblock("A", outgoing=["B"]),
                _Roadblock("B"),
            ]
        )
        route = self._build(fake_map, ["X", "A", "B"])
        self.assertEqual(route, ["X", "A", "B"])
        self.assertEqual(nu.route_connectivity_gaps(fake_map, route), [["X", "A"]])

    def test_oscillation_between_overlapping_connectors(self):
        # Mirrors 19003 -> 19061 -> 19003: a non-consecutive repeat across a junction.
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["B"]),
                _Roadblock("B"),
                _Roadblock("X"),
            ]
        )
        route = self._build(fake_map, ["A", "X", "A", "B"])
        self.assertEqual(route, ["A", "B"])

    def test_consecutive_duplicates_collapse(self):
        fake_map = _FakeMap([_Roadblock("A", outgoing=["B"]), _Roadblock("B")])
        self.assertEqual(self._build(fake_map, ["A", "A", "B", "B"]), ["A", "B"])

    def test_connectivity_gaps_empty_for_connected_route(self):
        fake_map = _FakeMap([_Roadblock("A", outgoing=["B"]), _Roadblock("B")])
        self.assertEqual(nu.route_connectivity_gaps(fake_map, ["A", "B"]), [])


class TestAppendGoalRoadblock(unittest.TestCase):
    def test_goal_already_last_is_untouched(self):
        fake_map = _FakeMap(
            [_Roadblock("A", outgoing=["B"]), _Roadblock("B", contains=[(5.0, 0.0)])]
        )
        route = nu._append_goal_roadblock(fake_map, ["A", "B"], _Point(5.0, 0.0))
        self.assertEqual(route, ["A", "B"])

    def test_dropped_goal_roadblock_is_appended_and_bridged(self):
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["B"]),
                _Roadblock("B", outgoing=["C"]),
                _Roadblock("C", contains=[(9.0, 0.0)]),
            ]
        )
        route = nu._append_goal_roadblock(fake_map, ["A"], _Point(9.0, 0.0))
        self.assertEqual(route, ["A", "B", "C"])

    def test_route_is_truncated_when_goal_lies_earlier(self):
        # The ego drove past the goal roadblock; the route must end at the goal, not beyond it.
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["B"]),
                _Roadblock("B", contains=[(5.0, 0.0)], outgoing=["C"]),
                _Roadblock("C"),
            ]
        )
        route = nu._append_goal_roadblock(fake_map, ["A", "B", "C"], _Point(5.0, 0.0))
        self.assertEqual(route, ["A", "B"])

    def _intersection_map(self):
        """Goal sits in an intersection: four roadblocks overlap on the same point.

        ``CROSS`` runs across the ego's direction and cannot be driven into from ``A``;
        ``ALONG`` runs with the ego and is reachable. Declared CROSS-first so that taking
        whatever the map query returned first would pick the wrong one.
        """
        goal = (5.0, 0.0)
        return _FakeMap(
            [
                _Roadblock("A", outgoing=["ALONG", "DIAG"]),
                _Roadblock("CROSS", contains=[goal], heading=math.pi / 2),
                _Roadblock("OPPOSED", contains=[goal], heading=math.pi),
                _Roadblock("ALONG", contains=[goal], heading=0.0),
                _Roadblock("DIAG", contains=[goal], heading=math.pi / 4),
            ]
        )

    def test_crossing_roadblock_is_not_chosen_for_a_goal_in_an_intersection(self):
        route = nu._append_goal_roadblock(self._intersection_map(), ["A"], _Point(5.0, 0.0), 0.0)
        self.assertEqual(route, ["A", "ALONG"])

    def test_reachability_outranks_heading(self):
        """A perfectly aligned roadblock the ego cannot drive into is still not the goal's."""
        goal = (5.0, 0.0)
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["DIAG"]),
                # Aligned with travel but unreachable from A.
                _Roadblock("ORPHAN", contains=[goal], heading=0.0),
                _Roadblock("DIAG", contains=[goal], heading=math.pi / 4),
            ]
        )
        route = nu._append_goal_roadblock(fake_map, ["A"], _Point(5.0, 0.0), 0.0)
        self.assertEqual(route, ["A", "DIAG"])

    def test_single_holder_is_used_without_a_heading(self):
        """Unambiguous goals keep the old behaviour, heading or not."""
        fake_map = _FakeMap(
            [_Roadblock("A", outgoing=["B"]), _Roadblock("B", contains=[(9.0, 0.0)])]
        )
        self.assertEqual(nu._append_goal_roadblock(fake_map, ["A"], _Point(9.0, 0.0)), ["A", "B"])

    def test_ambiguous_goal_without_heading_still_prefers_reachability(self):
        goal = (5.0, 0.0)
        fake_map = _FakeMap(
            [
                _Roadblock("A", outgoing=["REACHABLE"]),
                _Roadblock("ORPHAN", contains=[goal]),
                _Roadblock("REACHABLE", contains=[goal]),
            ]
        )
        route = nu._append_goal_roadblock(fake_map, ["A"], _Point(5.0, 0.0))
        self.assertEqual(route, ["A", "REACHABLE"])


class TestReplaceUnreachableStartRoadblock(unittest.TestCase):
    """The ego waits at a stop line inside both the through lane and the turn lane beside it."""

    def _fork_map(self):
        start = (0.0, 0.0)
        return _FakeMap(
            [
                # TURN is where the devkit committed; it leads away and never reaches THROUGH2.
                _Roadblock("TURN", contains=[start], outgoing=["TURN_EXIT"], heading=-0.6),
                _Roadblock("TURN_EXIT"),
                _Roadblock("STRAIGHT", contains=[start], outgoing=["THROUGH1"], heading=0.0),
                _Roadblock("THROUGH1", outgoing=["THROUGH2"]),
                _Roadblock("THROUGH2"),
            ]
        )

    def test_turn_lane_start_is_replaced_by_the_through_lane(self):
        route = nu._replace_unreachable_start_roadblock(
            self._fork_map(), ["TURN", "THROUGH2"], _Point(0.0, 0.0), 0.0
        )
        # STRAIGHT reaches THROUGH2 via THROUGH1, so the bridge is filled in too.
        self.assertEqual(route, ["STRAIGHT", "THROUGH1", "THROUGH2"])

    def test_connected_start_is_untouched(self):
        fake_map = _FakeMap(
            [
                _Roadblock("A", contains=[(0.0, 0.0)], outgoing=["B"], heading=0.0),
                _Roadblock("B"),
            ]
        )
        route = nu._replace_unreachable_start_roadblock(fake_map, ["A", "B"], _Point(0.0, 0.0), 0.0)
        self.assertEqual(route, ["A", "B"])

    def test_genuine_break_at_the_start_is_left_reported(self):
        """No candidate reaches the route, so the break is real and must stay visible."""
        fake_map = _FakeMap(
            [
                _Roadblock("ORPHAN", contains=[(0.0, 0.0)], heading=0.0),
                _Roadblock("ALSO_ORPHAN", contains=[(0.0, 0.0)], heading=0.1),
                _Roadblock("REST"),
            ]
        )
        route = nu._replace_unreachable_start_roadblock(
            fake_map, ["ORPHAN", "REST"], _Point(0.0, 0.0), 0.0
        )
        self.assertEqual(route, ["ORPHAN", "REST"])
        self.assertEqual(nu.route_connectivity_gaps(fake_map, route), [["ORPHAN", "REST"]])

    def test_candidate_aligned_with_the_ego_is_preferred(self):
        start = (0.0, 0.0)
        fake_map = _FakeMap(
            [
                _Roadblock("WRONG", contains=[start], outgoing=["NEXT"], heading=1.2),
                _Roadblock("ALIGNED", contains=[start], outgoing=["NEXT"], heading=0.02),
                _Roadblock("NEXT"),
            ]
        )
        route = nu._replace_unreachable_start_roadblock(
            fake_map, ["STRAY", "NEXT"], _Point(0.0, 0.0), 0.0
        )
        self.assertEqual(route, ["ALIGNED", "NEXT"])

    def test_single_element_route_is_untouched(self):
        self.assertEqual(
            nu._replace_unreachable_start_roadblock(
                self._fork_map(), ["TURN"], _Point(0.0, 0.0), 0.0
            ),
            ["TURN"],
        )


class TestTrajectoryArcLength(unittest.TestCase):
    def test_straight_line_length(self):
        states = [_State(0.0, 0.0), _State(3.0, 0.0), _State(3.0, 4.0)]
        self.assertAlmostEqual(nu.trajectory_arc_length_m(states), 7.0)

    def test_out_and_back_counts_the_return_leg(self):
        """The bug this replaces: an out-and-back measured only the outbound distance."""
        states = [_State(0.0, 0.0), _State(100.0, 0.0), _State(40.0, 0.0)]
        # Straight-line start->goal is only 40 m; the driven distance is 100 + 60.
        self.assertAlmostEqual(nu.trajectory_arc_length_m(states), 160.0)

    def test_single_state_is_zero(self):
        self.assertEqual(nu.trajectory_arc_length_m([_State(1.0, 1.0)]), 0.0)


class TestBuildRoutingTaskFromRoute(unittest.TestCase):
    """The task a live planner hands to select_route, built from an already-known route."""

    # A straight reference heading east, sampled every 10 m.
    REFERENCE = [(float(x), 0.0) for x in range(0, 101, 10)]

    # Stands in for the lane tangent project_goal_to_route_centerline reports at the goal.
    LANE_TANGENT = 0.3

    def _task(self, ego=(20.0, 1.0), goal=(80.0, 0.0), reference=None, **kwargs):
        with (
            mock.patch.object(
                nu, "project_start_to_route_midpoint", return_value=([20.0, 0.0], 0.0, "rb1")
            ),
            mock.patch.object(
                nu,
                "project_goal_to_route_centerline",
                return_value=([80.0, 0.0], self.LANE_TANGENT, "rb2"),
            ),
        ):
            return nu.build_routing_task_from_route(
                map_api=mock.Mock(),
                route_roadblock_ids=["rb1", "rb2"],
                ego_point=_Point(*ego),
                ego_heading=0.0,
                goal_xy=goal,
                map_name="us-ma-boston",
                epsg=32619,
                reference_polyline=self.REFERENCE if reference is None else reference,
                **kwargs,
            )

    def test_goal_equals_goal_proj_so_the_ladder_skips_rung_f(self):
        from route_description_generation.routing import route_attempts

        task = self._task()
        self.assertEqual(task["goal"], task["goal_proj"])
        attempts = route_attempts(
            start=task["start"],
            start_proj=task["start_proj"],
            goal=task["goal"],
            goal_proj=task["goal_proj"],
            vias=task["route_vias"],
        )
        self.assertEqual(len(attempts), 5)

    def test_reference_is_trimmed_to_the_stretch_from_ego_to_goal(self):
        task = self._task()
        self.assertEqual(task["route_polyline"], [(float(x), 0.0) for x in range(20, 81, 10)])
        self.assertAlmostEqual(task["nuplan_route_length_m"], 60.0)

    def test_length_is_measured_on_the_trimmed_path_not_the_whole_reference(self):
        """Otherwise the length check compares a 60 m remainder against a 100 m reference."""
        near_goal = self._task(ego=(70.0, 0.0))
        self.assertAlmostEqual(near_goal["nuplan_route_length_m"], 10.0)

    def test_arrival_heading_is_the_lane_tangent_not_the_direction_of_travel(self):
        """rdg sends the lane tangent to OSRM as the destination bearing; parity demands the same.

        The reference here runs due east (heading 0), so anything reading the heading off the
        driven path would return 0 rather than the lane's tangent.
        """
        self.assertAlmostEqual(self._task()["goal_heading"], self.LANE_TANGENT)

    def test_the_driven_path_seeds_the_lane_choice(self):
        """project_goal_to_route_centerline rejects lanes >90 deg from the heading it is given,
        so the seed is what keeps the goal off the oncoming carriageway."""
        turning = list(self.REFERENCE) + [(100.0, 10.0)]
        with (
            mock.patch.object(
                nu, "project_start_to_route_midpoint", return_value=([0.0, 0.0], 0.0, "rb1")
            ),
            mock.patch.object(
                nu, "project_goal_to_route_centerline", return_value=([100.0, 10.0], 1.0, "rb2")
            ) as m_goal,
        ):
            nu.build_routing_task_from_route(
                map_api=mock.Mock(),
                route_roadblock_ids=["rb1"],
                ego_point=_Point(0.0, 0.0),
                ego_heading=0.0,
                goal_xy=(100.0, 10.0),
                map_name="us-ma-boston",
                epsg=32619,
                reference_polyline=turning,
            )
        self.assertAlmostEqual(m_goal.call_args[0][3], math.pi / 2)

    def test_the_stored_goal_position_is_never_moved_by_the_heading_lookup(self):
        """The projected position is discarded: the caller's goal is taken as given."""
        with (
            mock.patch.object(
                nu, "project_start_to_route_midpoint", return_value=([0.0, 0.0], 0.0, "rb1")
            ),
            mock.patch.object(
                nu, "project_goal_to_route_centerline", return_value=([999.0, 999.0], 1.0, "rb2")
            ),
        ):
            task = nu.build_routing_task_from_route(
                map_api=mock.Mock(),
                route_roadblock_ids=["rb1"],
                ego_point=_Point(0.0, 0.0),
                ego_heading=0.0,
                goal_xy=(80.0, 0.0),
                map_name="us-ma-boston",
                epsg=32619,
                reference_polyline=self.REFERENCE,
            )
        self.assertEqual(task["goal"], (80.0, 0.0))
        self.assertEqual(task["goal_proj"], (80.0, 0.0))

    def test_a_stopped_ego_still_seeds_from_its_last_real_movement(self):
        """Repeated vertices are walked past rather than read as a zero-length final segment."""
        stalled = self.REFERENCE + [(100.0, 0.0), (100.0, 0.0)]
        self.assertAlmostEqual(
            self._task(goal=(100.0, 0.0), reference=stalled)["goal_heading"], self.LANE_TANGENT
        )

    def test_stationary_jitter_does_not_reverse_the_seed_heading(self):
        """A stopped ego logs sub-millimetre wobble whose direction is noise.

        Reading the last segment merely longer than zero once seeded the lane choice with a heading
        up to 180 deg out, so project_goal_to_route_centerline picked the oncoming carriageway and
        OSRM answered with a U-turn — a 202 m route came back as 2777 m.
        """
        # Heading east, then parked, with the final wobble pointing back west.
        jittered = [(0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (100.0002, 0.0), (100.0, 0.0)]
        self.assertAlmostEqual(nu._polyline_end_heading(jittered), 0.0)

    def test_a_short_real_movement_is_still_trusted(self):
        """The span threshold must not reject genuine slow motion as jitter."""
        crawling = [(0.0, 0.0), (0.0, 1.5)]
        self.assertAlmostEqual(nu._polyline_end_heading(crawling), math.pi / 2)

    def test_a_reference_that_never_moves_leaves_the_arrival_unconstrained(self):
        """With no direction to disambiguate lanes, an unconstrained arrival beats a guessed one."""
        self.assertIsNone(self._task(reference=[(20.0, 0.0)] * 4)["goal_heading"])

    def test_raw_ego_position_is_the_unprojected_start(self):
        task = self._task(ego=(20.0, 1.0))
        self.assertEqual(task["start"], (20.0, 1.0))
        self.assertEqual(task["start_proj"], (20.0, 0.0))

    def test_vias_are_passed_through_untouched(self):
        vias = [(50.0, 0.0, 0.0)]
        self.assertEqual(self._task(vias=vias)["route_vias"], vias)
        self.assertIsNone(self._task()["route_vias"])

    def test_task_carries_every_key_select_route_reads(self):
        task = self._task()
        for key in (
            "start",
            "start_proj",
            "goal",
            "goal_proj",
            "map_name",
            "epsg",
            "start_heading",
            "goal_heading",
            "route_vias",
            "route_polyline",
            "nuplan_route_length_m",
        ):
            self.assertIn(key, task)


if __name__ == "__main__":
    unittest.main()
