import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse

from route_description_generation import osrm_client as od


def _capture_url(**kwargs):
    """Call request_route with a stubbed transport and return the requested URL."""
    with mock.patch.object(od.requests, "get") as m_get:
        m_get.return_value.json.return_value = {"code": "Ok"}
        od.request_route(**kwargs)
    return m_get.call_args[0][0]


def _bearings(url):
    qs = parse_qs(urlparse(url).query)
    return qs["bearings"][0].split(";") if "bearings" in qs else None


class TestOsrmViaBearings(unittest.TestCase):
    def test_via_bearings_are_emitted_per_waypoint(self):
        url = _capture_url(
            start=(42.0, -71.0),
            destination=(42.01, -71.01),
            server_url="http://localhost:5001",
            start_bearing=90.0,
            goal_bearing=270.0,
            bearing_tol=45.0,
            vias=[(42.005, -71.005, 180.0), (42.007, -71.007, 0.0)],
        )
        entries = _bearings(url)
        # OSRM requires exactly one entry per coordinate: start + 2 vias + destination.
        self.assertEqual(len(entries), 4)
        self.assertEqual(entries, ["90,45", "180,45", "0,45", "270,45"])

    def test_via_without_heading_stays_unconstrained(self):
        url = _capture_url(
            start=(42.0, -71.0),
            destination=(42.01, -71.01),
            server_url="http://localhost:5001",
            start_bearing=90.0,
            vias=[(42.005, -71.005, None)],
        )
        self.assertEqual(_bearings(url), ["90,45", "", ""])

    def test_via_bearings_alone_still_emit_the_parameter(self):
        # No start/goal bearing, but a via has one -> bearings must still be sent.
        url = _capture_url(
            start=(42.0, -71.0),
            destination=(42.01, -71.01),
            server_url="http://localhost:5001",
            vias=[(42.005, -71.005, 123.4)],
        )
        self.assertEqual(_bearings(url), ["", "123,45", ""])

    def test_two_point_request_has_no_bearings_when_unconstrained(self):
        url = _capture_url(
            start=(42.0, -71.0),
            destination=(42.01, -71.01),
            server_url="http://localhost:5001",
        )
        self.assertIsNone(_bearings(url))

    def test_bearings_are_wrapped_into_range(self):
        url = _capture_url(
            start=(42.0, -71.0),
            destination=(42.01, -71.01),
            server_url="http://localhost:5001",
            vias=[(42.005, -71.005, 361.0)],
        )
        self.assertEqual(_bearings(url), ["", "1,45", ""])


class TestNonOkResponsesAreTolerated(unittest.TestCase):
    """A bearing OSRM cannot satisfy yields NoSegment; the ladder must fall through, not raise."""

    NO_SEGMENT = {"code": "NoSegment", "message": "Could not find a matching segment"}

    def test_routing_to_language_returns_empty(self):
        self.assertEqual(od.route_to_instructions(self.NO_SEGMENT), "")

    def test_maneuver_positions_are_padded_empty(self):
        positions = od.extract_maneuver_positions(self.NO_SEGMENT, epsg=32619, max_maneuvers=5)
        self.assertEqual(len(positions), 5)

    def test_first_turn_distance_is_none(self):
        self.assertIsNone(od.first_turn_distance_m(self.NO_SEGMENT))


if __name__ == "__main__":
    unittest.main()
