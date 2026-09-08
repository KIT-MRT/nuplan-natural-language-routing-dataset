import math
import unittest

from route_description_generation.geometry import (
    decode_polyline,
    path_deviation,
    polyline_length_m,
    resample_polyline,
    trim_polyline,
)


class TestDecodePolyline(unittest.TestCase):
    def test_reference_value_from_the_polyline_spec(self):
        """The canonical example from Google's encoded-polyline documentation."""
        decoded = decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@")
        expected = [(-120.2, 38.5), (-120.95, 40.7), (-126.453, 43.252)]
        self.assertEqual(len(decoded), len(expected))
        for (lon, lat), (elon, elat) in zip(decoded, expected):
            self.assertAlmostEqual(lon, elon, places=5)
            self.assertAlmostEqual(lat, elat, places=5)

    def test_empty_string_decodes_to_no_points(self):
        self.assertEqual(decode_polyline(""), [])

    def test_deltas_accumulate(self):
        """Each pair is a delta on the previous, so a repeated delta walks in a straight line."""
        points = decode_polyline("_ibE_ibE_ibE_ibE")
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[1][0] - points[0][0], points[0][0], places=5)


class TestResamplePolyline(unittest.TestCase):
    def test_straight_line_is_evenly_spaced(self):
        out = resample_polyline([(0.0, 0.0), (20.0, 0.0)], 5.0)
        self.assertEqual(out, [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (15.0, 0.0), (20.0, 0.0)])

    def test_endpoints_are_always_kept(self):
        out = resample_polyline([(0.0, 0.0), (12.0, 0.0)], 5.0)
        self.assertEqual(out[0], (0.0, 0.0))
        self.assertEqual(out[-1], (12.0, 0.0))

    def test_original_vertices_are_preserved(self):
        """Dropping a mid-stride vertex would cut the corner it represents."""
        out = resample_polyline([(0.0, 0.0), (6.0, 0.0), (12.0, 0.0)], 5.0)
        xs = [round(x, 6) for x, _ in out]
        self.assertIn(6.0, xs)
        self.assertEqual(xs, [0.0, 3.0, 6.0, 9.0, 12.0])

    def test_no_gap_exceeds_the_spacing(self):
        out = resample_polyline([(0.0, 0.0), (0.0, 13.0), (7.0, 13.0)], 5.0)
        gaps = [math.dist(p, q) for p, q in zip(out, out[1:])]
        self.assertLessEqual(max(gaps), 5.0 + 1e-9)
        self.assertIn((0.0, 13.0), out)

    def test_short_input_is_returned_unchanged(self):
        self.assertEqual(resample_polyline([(1.0, 2.0)], 5.0), [(1.0, 2.0)])
        self.assertEqual(resample_polyline([], 5.0), [])

    def test_repeated_points_do_not_stall(self):
        out = resample_polyline([(0.0, 0.0), (0.0, 0.0), (10.0, 0.0)], 5.0)
        self.assertEqual(out[0], (0.0, 0.0))
        self.assertEqual(out[-1], (10.0, 0.0))


class TestPathDeviation(unittest.TestCase):
    def test_identical_paths_have_zero_deviation(self):
        path = [(0.0, 0.0), (100.0, 0.0)]
        result = path_deviation(path, path)
        assert result is not None
        self.assertAlmostEqual(result["max_m"], 0.0, places=6)
        self.assertAlmostEqual(result["p95_m"], 0.0, places=6)

    def test_parallel_offset_paths_report_the_offset(self):
        a = [(0.0, 0.0), (100.0, 0.0)]
        b = [(0.0, 3.0), (100.0, 3.0)]
        result = path_deviation(a, b)
        assert result is not None
        self.assertAlmostEqual(result["max_m"], 3.0, places=6)
        self.assertAlmostEqual(result["mean_m"], 3.0, places=6)

    def test_sparse_vertices_still_match_a_dense_path(self):
        """Segment distance, not vertex distance: two vertices must match 21 collinear ones."""
        sparse = [(0.0, 0.0), (100.0, 0.0)]
        dense = [(float(i) * 5.0, 0.0) for i in range(21)]
        result = path_deviation(sparse, dense)
        assert result is not None
        self.assertAlmostEqual(result["max_m"], 0.0, places=6)

    def test_mean_is_not_diluted_by_the_denser_path(self):
        """Pooling would let the densely-sampled side outvote the other purely on point count."""
        # 'driven' is sampled far more finely than 'route', and lies entirely on it, so the
        # driven->route direction is ~0. The route's own excursion must still show up.
        driven = [(float(i) * 0.5, 0.0) for i in range(41)]
        route = [(0.0, 0.0), (10.0, 40.0), (20.0, 0.0)]
        result = path_deviation(driven, route)
        assert result is not None
        self.assertGreater(result["mean_m"], 10.0)

    def test_detour_is_caught(self):
        driven = [(0.0, 0.0), (100.0, 0.0)]
        detour = [(0.0, 0.0), (50.0, 60.0), (100.0, 0.0)]
        result = path_deviation(driven, detour)
        assert result is not None
        self.assertGreater(result["max_m"], 50.0)

    def test_skipped_section_is_caught_by_the_reverse_direction(self):
        """A route that omits a leg the ego drove: only b->a sees it, so symmetry matters."""
        driven = [(0.0, 0.0), (0.0, 100.0), (100.0, 100.0)]
        shortcut = [(0.0, 0.0), (0.0, 5.0)]
        result = path_deviation(driven, shortcut)
        assert result is not None
        self.assertGreater(result["max_m"], 90.0)

    def test_doubling_back_on_itself_is_not_penalised(self):
        """The ego drives out and returns; the route covers the same road once."""
        out_and_back = [(0.0, 0.0), (100.0, 0.0), (0.0, 0.0)]
        one_way = [(0.0, 0.0), (100.0, 0.0)]
        result = path_deviation(out_and_back, one_way)
        assert result is not None
        self.assertAlmostEqual(result["max_m"], 0.0, places=6)

    def test_empty_path_returns_none(self):
        self.assertIsNone(path_deviation([], [(0.0, 0.0), (1.0, 0.0)]))


class TestPolylineLength(unittest.TestCase):
    def test_sums_segment_lengths(self):
        self.assertAlmostEqual(polyline_length_m([(0.0, 0.0), (3.0, 4.0), (3.0, 9.0)]), 10.0)

    def test_degenerate_inputs_are_zero_length(self):
        self.assertEqual(polyline_length_m([]), 0.0)
        self.assertEqual(polyline_length_m([(1.0, 2.0)]), 0.0)

    def test_agrees_with_trajectory_arc_length_on_the_same_points(self):
        """route_extraction.trajectory_arc_length_m delegates here; the two must not drift apart."""
        from route_description_generation.route_extraction import trajectory_arc_length_m

        class _Point:
            def __init__(self, x, y):
                self.x, self.y = x, y

        class _State:
            def __init__(self, x, y):
                self.rear_axle = type("RA", (), {"point": _Point(x, y)})()

        points = [(0.0, 0.0), (3.0, 4.0), (3.0, 9.0), (-1.0, 9.0)]
        states = [_State(x, y) for x, y in points]
        self.assertAlmostEqual(trajectory_arc_length_m(states), polyline_length_m(points))


class TestTrimPolyline(unittest.TestCase):
    LINE = [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (30.0, 0.0), (40.0, 0.0)]

    def test_cuts_at_the_nearest_vertex_to_each_end(self):
        self.assertEqual(
            trim_polyline(self.LINE, start_xy=(9.0, 2.0), goal_xy=(31.0, -1.0)),
            [(10.0, 0.0), (20.0, 0.0), (30.0, 0.0)],
        )

    def test_omitted_ends_are_left_alone(self):
        self.assertEqual(trim_polyline(self.LINE, start_xy=(19.0, 0.0)), self.LINE[2:])
        self.assertEqual(trim_polyline(self.LINE, goal_xy=(21.0, 0.0)), self.LINE[:3])
        self.assertEqual(trim_polyline(self.LINE), self.LINE)

    def test_result_is_a_contiguous_ordered_subsequence(self):
        """resample_polyline relies on every original vertex surviving, in order."""
        trimmed = trim_polyline(self.LINE, start_xy=(11.0, 0.0), goal_xy=(29.0, 0.0))
        self.assertTrue(all(p in self.LINE for p in trimmed))
        self.assertEqual(trimmed, sorted(trimmed, key=self.LINE.index))

    def test_a_goal_behind_the_start_collapses_instead_of_reversing(self):
        trimmed = trim_polyline(self.LINE, start_xy=(30.0, 0.0), goal_xy=(0.0, 0.0))
        self.assertEqual(trimmed, [(30.0, 0.0)])
        self.assertEqual(polyline_length_m(trimmed), 0.0)

    def test_empty_input_stays_empty(self):
        self.assertEqual(trim_polyline([], start_xy=(0.0, 0.0), goal_xy=(1.0, 0.0)), [])


if __name__ == "__main__":
    unittest.main()
