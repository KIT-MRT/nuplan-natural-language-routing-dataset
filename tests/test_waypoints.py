import unittest

from route_description_generation.waypoints import (
    MAX_ROUTE_VIAS,
    filter_ahead_vias,
    select_spread_vias,
    truncate_waypoints_before_goal,
)


class TestRouteWaypointUtils(unittest.TestCase):
    def test_filter_ahead_vias_drops_none_and_applies_spacing(self):
        waypoints = [
            (0.0, 0.0),
            None,
            (1.0, 1.0),  # too close to previous kept point when spacing=2.0
            (4.0, 4.0),
        ]
        filtered = filter_ahead_vias(waypoints, k=-1, min_spacing_m=2.0)
        self.assertEqual(filtered, [(0.0, 0.0), (4.0, 4.0)])

    def test_filter_ahead_vias_respects_cursor(self):
        waypoints = [
            (0.0, 0.0),
            (5.0, 0.0),
            (10.0, 0.0),
        ]
        filtered = filter_ahead_vias(waypoints, k=0, min_spacing_m=0.1)
        self.assertEqual(filtered, [(5.0, 0.0), (10.0, 0.0)])

    def test_truncate_waypoints_before_goal_index(self):
        waypoints = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0)]
        truncated = truncate_waypoints_before_goal(waypoints, goal_route_index=2)
        self.assertEqual(truncated, [(0.0, 0.0), (5.0, 0.0)])

    def test_truncate_waypoints_fallback_when_goal_unknown(self):
        waypoints = [(0.0, 0.0), (5.0, 0.0)]
        truncated = truncate_waypoints_before_goal(waypoints, goal_route_index=None)
        self.assertEqual(truncated, waypoints)


class TestSelectSpreadVias(unittest.TestCase):
    """Few, well-spread vias: each extra one is another chance for OSRM to snap wrongly."""

    @staticmethod
    def _c(x, length):
        # (x, y, heading, roadblock_length)
        return (float(x), 0.0, 0.0, float(length))

    def test_fewer_than_cap_passes_through_without_length(self):
        cands = [self._c(0, 50), self._c(10, 40)]
        self.assertEqual(
            select_spread_vias(cands, max_vias=3),
            [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        )

    def test_caps_and_spreads_along_the_route(self):
        cands = [self._c(i, 10) for i in range(9)]
        chosen = select_spread_vias(cands, max_vias=3)
        self.assertEqual(len(chosen), 3)
        # One from each third, in route order.
        xs = [c[0] for c in chosen]
        self.assertEqual(xs, sorted(xs))
        self.assertLess(xs[0], 3)
        self.assertTrue(3 <= xs[1] < 6)
        self.assertGreaterEqual(xs[2], 6)

    def test_prefers_the_longest_roadblock_in_each_bucket(self):
        # Bucket 1: x=0..2, longest is x=1. Bucket 2: x=3..5, longest is x=5.
        cands = [
            self._c(0, 12),
            self._c(1, 99),
            self._c(2, 15),
            self._c(3, 11),
            self._c(4, 20),
            self._c(5, 80),
        ]
        chosen = select_spread_vias(cands, max_vias=2)
        self.assertEqual([c[0] for c in chosen], [1.0, 5.0])

    def test_never_exceeds_the_default_cap(self):
        cands = [self._c(i, 10) for i in range(19)]
        self.assertEqual(len(select_spread_vias(cands)), MAX_ROUTE_VIAS)

    def test_empty_returns_none(self):
        self.assertIsNone(select_spread_vias([]))


if __name__ == "__main__":
    unittest.main()
