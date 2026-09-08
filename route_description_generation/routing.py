"""Requesting routes from OSRM and checking them against the driven route.

`process_routing_task` walks a ladder of OSRM request variants and keeps the first that agrees
with the ego's driven route on both length and shape; see its comments for what each rung exists
for.
"""

import math
import random
import time
from multiprocessing import Semaphore
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, cast

import requests
import urllib3
from pyproj import Geod
from pyproj import Transformer as ProjTransformer

from route_description_generation.geometry import (
    decode_polyline,
    path_deviation,
)
from route_description_generation.osrm_client import (
    extract_maneuver_positions,
    first_turn_distance_m,
    request_route,
    route_to_instructions,
)
from route_description_generation.route_extraction import (
    build_routing_task_from_alternative_route,
    build_routing_task_from_scenario,
)

# Geodesic helper for converting a map-frame heading into a true-north compass bearing.
_GEOD = Geod(ellps="WGS84")

# If using OSRM: Routing server have to be setup and running locally!
OSRM_PORT_US_NOTHEAST = 5001
OSRM_PORT_US_WEST = 5002
OSRM_PORT_SINGAPORE = 5003

# Limit number of concurrent OSRM requests: OSRM server can handle only a few at a time
MAX_CONCURRENT_REQUESTS = 2
semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

# Exceptions that trigger a retry for OSRM requests
RETRY_EXCEPTIONS = (
    ConnectionResetError,
    ConnectionAbortedError,
    OSError,
    requests.exceptions.RequestException,
    urllib3.exceptions.HTTPError,
)

ROUTE_LENGTH_REL_TOLERANCE = 0.15
ROUTE_LENGTH_REL_TOLERANCE_MEDIUM = 0.2  # for nuPlan route length <= 150 m
ROUTE_LENGTH_REL_TOLERANCE_SHORT = 0.3  # for nuPlan route length <= 100 m

# Allowed deviation (degrees) between a waypoint's travel bearing and the bearing of the OSM way
# OSRM may snap it to. Kept well below 90 so the oncoming carriageway (180 deg off) stays excluded.
BEARING_TOL_DEG = 45.0
# Relaxed tolerance for the last-resort attempt. When the ego is mid-turn its instantaneous heading
# can sit more than 45 deg away from *every* nearby OSM way, because OSM models a curve as a couple
# of coarse straight segments. OSRM then rejects the ways under the ego and snaps to the first
# matching one further down the road - tens of metres away - which truncates the route's start and
# blows the length check. Widening the cone lets the way the ego is actually on qualify again.
BEARING_TOL_WIDE_DEG = 60.0

# p95 deviation (metres) from the driven path below which a length-agreeing attempt is accepted
# immediately, without trying the remaining rungs. Measured over val14: valid routes sit at a
# median p95 of 4.6 m and a p99 of 14.1 m, so 20 m keeps the ladder short-circuiting on essentially
# every well-behaved scenario while still exploring the handful where an early rung agrees on
# length only marginally and a later rung follows the driven path far more closely.
PATH_DEVIATION_ACCEPT_P95_M = 20.0

# Mean deviation (metres) from the driven path above which a route is rejected however well its
# length agrees. Length agreement alone lets through routes down an entirely different street.
PATH_DEVIATION_MAX_MEAN_M = 12.0

# The same gate when the reference is a lane centerline rather than a driven trajectory (no ground
# truth exists, e.g. alternative routing).
PATH_DEVIATION_MAX_P95_CENTERLINE_M = 30.0

# Which deviation statistic gates a route, keyed by what the reference path was derived from.
PATH_CHECKS = {
    "trajectory": ("mean", PATH_DEVIATION_MAX_MEAN_M),
    "centerline": ("p95", PATH_DEVIATION_MAX_P95_CENTERLINE_M),
}


def validate_nuplan_vs_osrm_route(
    routing_output: Dict,
    nuplan_route_length_m: Optional[float],
    tolerance: float = ROUTE_LENGTH_REL_TOLERANCE,
) -> Dict[str, Optional[float]]:
    """Check whether OSRM route distance agrees with nuPlan route length within tolerance."""
    routes = (routing_output or {}).get("routes") or []
    osrm_distance = None
    if routes and isinstance(routes[0], dict):
        distance = routes[0].get("distance")
        if isinstance(distance, (int, float)):
            osrm_distance = float(distance)

    valid = False
    ratio = None
    if isinstance(nuplan_route_length_m, (int, float)) and isinstance(osrm_distance, float):
        ratio = float(nuplan_route_length_m) / max(osrm_distance, 1e-6)
        turn_allowance_m = first_turn_distance_m(routing_output)
        if 0 <= nuplan_route_length_m < 10 and 0 <= osrm_distance < 10:
            # Both routes are very short, so we accept them as valid regardless of ratio.
            valid = True
        elif nuplan_route_length_m <= 100:
            # enlarge tolerance for short routes
            valid = (
                (1.0 - ROUTE_LENGTH_REL_TOLERANCE_SHORT)
                <= ratio
                <= (1.0 + ROUTE_LENGTH_REL_TOLERANCE_SHORT)
            )
        elif nuplan_route_length_m <= 150:
            # enlarge tolerance for medium-length routes
            valid = (
                (1.0 - ROUTE_LENGTH_REL_TOLERANCE_MEDIUM)
                <= ratio
                <= (1.0 + ROUTE_LENGTH_REL_TOLERANCE_MEDIUM)
            )
        else:
            valid = (1.0 - tolerance) <= ratio <= (1.0 + tolerance)

        # Additional condition for routes with turns as next step:
        # OSRM calculates turn distance to intersection midpoint
        if not valid and turn_allowance_m is not None:
            valid = (
                (1.0 - tolerance) <= ratio <= (1.0 + tolerance)
                or (1.0 - tolerance)
                <= (float(nuplan_route_length_m) / max(osrm_distance - turn_allowance_m, 1e-6))
                <= (1.0 + tolerance)
                or (osrm_distance - nuplan_route_length_m <= turn_allowance_m)
            )

    return {
        "valid": bool(valid),
        "nuplan_length_m": (
            float(nuplan_route_length_m)
            if isinstance(nuplan_route_length_m, (int, float))
            else None
        ),
        "osrm_length_m": osrm_distance,
        "ratio_nuplan_over_osrm": ratio,
        "relative_tolerance": float(tolerance),
    }


def routing(
    start: tuple,
    destination: tuple,
    map_name: str,
    cs: Optional[str] = None,
    routing_service: str = "osrm",
    start_heading: Optional[float] = None,
    goal_heading: Optional[float] = None,
    bearing_tol: float = BEARING_TOL_DEG,
    vias=None,
):
    """
    Get driving directions between start and destination using OSRM.
    Args:
        start (tuple): A tuple of (latitude, longitude) for the starting point.
        destination (tuple): A tuple of (latitude, longitude) for the destination point.
        map_name (str): Name of the map.
        cs (str, optional): Coordinate system of the input points (e.g., "EPSG:32633")
                            -> only required if the start and destination input is not lat/lon.
        routing_service (str): Routing service to use. Only ``"osrm"`` is supported.
        start_heading (float, optional): Departure heading at ``start`` in the ``cs`` map frame
            (radians, CCW from +x). When given (and ``cs`` is set, OSRM only), it is converted to a
            true-north compass bearing and passed to OSRM so the query start snaps to the correct
            travel direction instead of the oncoming carriageway. Requires ``cs`` to be set.
        goal_heading (float, optional): Arrival heading at ``destination`` in the ``cs`` map frame
            (radians, CCW from +x). Same handling as ``start_heading`` but constrains OSRM's
            arrival direction at the destination instead of the departure. Requires ``cs``.
        bearing_tol (float): Allowed deviation (degrees) around the derived bearings.
        vias (sequence of (x, y), optional): Intermediate waypoints (in the same ``cs`` frame as
            ``start``/``destination``) the OSRM route must pass through, in order. Transformed to
            lat/lon alongside start/destination and forwarded to OSRM. ``None`` = plain two-point
            request (unchanged). OSRM only.
    Returns:
        dict: The routing information returned by specified routing service.
    """
    start_bearing = None
    goal_bearing = None
    if cs is not None:
        transformer = ProjTransformer.from_crs(f"{cs}", "EPSG:4326", always_xy=True)
        lon_start, lat_start = transformer.transform(start[0], start[1])
        if start_heading is not None:
            look_m = 5.0
            ahead_x = start[0] + math.cos(start_heading) * look_m
            ahead_y = start[1] + math.sin(start_heading) * look_m
            lon_ahead, lat_ahead = transformer.transform(ahead_x, ahead_y)
            fwd_az, _, _ = _GEOD.inv(lon_start, lat_start, lon_ahead, lat_ahead)
            start_bearing = fwd_az % 360.0
        start = (lat_start, lon_start)
        lon_dest, lat_dest = transformer.transform(destination[0], destination[1])
        if goal_heading is not None:
            look_m = 5.0
            ahead_x = destination[0] + math.cos(goal_heading) * look_m
            ahead_y = destination[1] + math.sin(goal_heading) * look_m
            lon_ahead, lat_ahead = transformer.transform(ahead_x, ahead_y)
            fwd_az, _, _ = _GEOD.inv(lon_dest, lat_dest, lon_ahead, lat_ahead)
            goal_bearing = fwd_az % 360.0
        destination = (lat_dest, lon_dest)
        if vias:
            # Each via carries a heading; convert it to a true-north bearing the same way as
            # start/goal so OSRM snaps the via to a way travelling that direction rather than to
            # the oncoming carriageway or a crossing street.
            transformed_vias = []
            for via in vias:
                vx, vy = via[0], via[1]
                lon_v, lat_v = transformer.transform(vx, vy)
                via_bearing = None
                if len(via) > 2 and via[2] is not None:
                    look_m = 5.0
                    ahead_x = vx + math.cos(via[2]) * look_m
                    ahead_y = vy + math.sin(via[2]) * look_m
                    lon_ahead, lat_ahead = transformer.transform(ahead_x, ahead_y)
                    fwd_az, _, _ = _GEOD.inv(lon_v, lat_v, lon_ahead, lat_ahead)
                    via_bearing = fwd_az % 360.0
                transformed_vias.append((lat_v, lon_v, via_bearing))
            vias = transformed_vias

    if routing_service == "osrm":
        if map_name == "us-ma-boston" or map_name == "us-pa-pittsburgh-hazelwood":
            osrm_port = OSRM_PORT_US_NOTHEAST
        elif map_name == "us-nv-las-vegas-strip":
            osrm_port = OSRM_PORT_US_WEST
        elif map_name == "sg-one-north":
            osrm_port = OSRM_PORT_SINGAPORE
        else:
            raise ValueError(f"Unknown map_name for OSRM routing: {map_name!r}")
        server_url = f"http://localhost:{osrm_port}"
        request_route_fn = cast(Any, request_route)
        return request_route_fn(
            start,
            destination,
            server_url,
            start_bearing=start_bearing,
            goal_bearing=goal_bearing,
            bearing_tol=bearing_tol,
            vias=vias,
        )
    raise ValueError(f"Unsupported routing_service: {routing_service!r}")


def routing_to_language(routing_directions, routing_service="osrm"):
    """
    Convert routing information to a natural language description.
    Args:
        routing_info (dict): The routing information from the routing service.
        routing_service (str): The routing service used. Only ``"osrm"`` is supported.
    Returns:
        str: A natural language description of the route.
    """
    if routing_service != "osrm":
        raise ValueError(f"Unsupported routing_service: {routing_service!r}")
    return route_to_instructions(routing_directions)


def request_route_with_retry(**kwargs):
    """
    Call the routing function with retries on failure.
    """
    delay = 0.25
    for attempt in range(6):
        try:
            return routing(**kwargs)
        except RETRY_EXCEPTIONS:
            if attempt == 5:
                raise
            time.sleep(delay)
            delay *= 2


def execute_routing_request(
    start_abs: tuple,
    goal_abs: tuple,
    map_name: str,
    epsg: int,
    routing_service: str = "osrm",
    start_heading: Optional[float] = None,
    goal_heading: Optional[float] = None,
    vias=None,
    bearing_tol: float = BEARING_TOL_DEG,
) -> tuple:
    with semaphore:
        # Micro-jitter: prevents synchronized requests from multiple workers
        time.sleep(random.uniform(0.0, 0.02))
        routing_output = request_route_with_retry(
            start=start_abs,
            destination=goal_abs,
            map_name=map_name,
            cs=f"EPSG:{epsg}",
            routing_service=routing_service,
            start_heading=start_heading,
            goal_heading=goal_heading,
            bearing_tol=bearing_tol,
            vias=vias,
        )

    lg_description = routing_to_language(routing_output, routing_service=routing_service)

    if routing_service == "osrm":
        maneuver_positions = extract_maneuver_positions(routing_output, epsg)
    else:
        maneuver_positions = []
    # Add other routing services here if needed
    return routing_output, lg_description, maneuver_positions


def execute_and_validate_routing_request(
    start_abs: tuple,
    goal_abs: tuple,
    map_name: str,
    epsg: int,
    routing_service: str,
    start_heading: Optional[float],
    goal_heading: Optional[float],
    vias,
    nuplan_route_length_m: Optional[float],
    bearing_tol: float = BEARING_TOL_DEG,
    route_polyline=None,
    path_check: tuple = PATH_CHECKS["trajectory"],
) -> tuple:
    """Run one routing request and check it against the nuPlan route length and geometry.

    ``path_check`` is the ``(statistic, limit)`` pair gating the shape agreement; it depends on what
    the reference path was derived from (see ``PATH_CHECKS``).

    Returns ``(routing_output, lg_description, maneuver_positions, route_validation)``.
    """
    routing_output, lg_description, maneuver_positions = execute_routing_request(
        start_abs,
        goal_abs,
        map_name,
        epsg,
        routing_service=routing_service,
        start_heading=start_heading,
        goal_heading=goal_heading,
        vias=vias,
        bearing_tol=bearing_tol,
    )
    route_validation = validate_nuplan_vs_osrm_route(
        routing_output,
        nuplan_route_length_m,
    )
    route_validation.update(osrm_path_deviation(routing_output, epsg, route_polyline))

    # A route counts as valid only when it agrees with the reference on both length and shape.
    # Both components stay in the record so a rejection can be attributed to one or the other.
    statistic, limit = path_check
    deviation = route_validation.get(f"path_deviation_{statistic}_m")
    route_validation["length_valid"] = bool(route_validation["valid"])
    route_validation["path_valid"] = deviation is None or deviation <= limit
    route_validation["valid"] = bool(
        route_validation["length_valid"] and route_validation["path_valid"]
    )
    return routing_output, lg_description, maneuver_positions, route_validation


def osrm_path_deviation(
    routing_output: Optional[Dict], epsg: int, route_polyline
) -> Dict[str, Optional[float]]:
    """Deviation between OSRM's returned geometry and the path the ego actually drove.

    Recorded alongside the length ratio because the two disagree in both directions: a route can
    match in length while describing a different street, and can follow the driven path closely
    while disagreeing on length. ``None`` whenever either geometry is unavailable.
    """
    empty: Dict[str, Optional[float]] = {
        "path_deviation_max_m": None,
        "path_deviation_p95_m": None,
        "path_deviation_mean_m": None,
    }
    routes = (routing_output or {}).get("routes") or []
    geometry = routes[0].get("geometry") if routes else None
    if not geometry or not route_polyline:
        return empty

    transformer = ProjTransformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    osrm_path = [transformer.transform(lon, lat) for lon, lat in decode_polyline(geometry)]
    deviation = path_deviation(osrm_path, [tuple(p) for p in route_polyline])
    if deviation is None:
        return empty
    return {
        "path_deviation_max_m": deviation["max_m"],
        "path_deviation_p95_m": deviation["p95_m"],
        "path_deviation_mean_m": deviation["mean_m"],
    }


class RouteAttempt(NamedTuple):
    """One rung of the request ladder: a set of OSRM request parameters to try."""

    start: tuple
    goal: tuple
    vias: Optional[Sequence]
    bearing_tol: float
    strategy: str


class RouteSelection(NamedTuple):
    """The attempt that won the ladder, with everything needed to describe or record it."""

    routing_output: Optional[Dict]
    description: Optional[str]
    maneuver_positions: list
    validation: Dict
    strategy: Optional[str]
    start: tuple
    goal: tuple


def route_attempts(*, start, start_proj, goal, goal_proj, vias) -> List[RouteAttempt]:
    """The ladder of OSRM request variants, in the order they are tried.

    A/B: vias with the projected then raw start. C/D: the same two without vias. E repeats A with a
    widened bearing cone, for an ego mid-turn where no OSM way beneath it matches its heading within
    45 deg. F uses both endpoints unprojected, for when projecting lands them either side of an OSM
    seam and forces a detour between points metres apart.

    F is skipped when ``goal == goal_proj`` (a caller with a fixed goal), where it would send exactly
    D's request.
    """
    # fmt: off
    ladder = [
        RouteAttempt(start_proj, goal_proj, vias, BEARING_TOL_DEG, "A_vias_start_proj"),
        RouteAttempt(start, goal_proj, vias, BEARING_TOL_DEG, "B_vias_start"),
        RouteAttempt(start_proj, goal_proj, None, BEARING_TOL_DEG, "C_no_vias_start_proj"),
        RouteAttempt(start, goal_proj, None, BEARING_TOL_DEG, "D_no_vias_start"),
        RouteAttempt(start_proj, goal_proj, vias, BEARING_TOL_WIDE_DEG, "E_vias_start_proj_wide_bearing"),
    ]
    # fmt: on
    if tuple(goal) != tuple(goal_proj):
        ladder.append(
            RouteAttempt(start, goal, None, BEARING_TOL_DEG, "F_no_vias_raw_start_and_goal")
        )
    return ladder


def select_route(routing_task: Dict, routing_service: str = "osrm") -> RouteSelection:
    """Walk the request ladder and return the attempt that best matches the reference route.

    An attempt is accepted only when it agrees on **both** length and shape, and short-circuits the
    remaining rungs only when it also comes within ``PATH_DEVIATION_ACCEPT_P95_M`` of the driven
    path. Otherwise the walk continues and the accepted attempt closest to that path wins: stopping
    at the first length-agreeing rung used to keep routes that passed the ratio check only
    marginally while a later rung followed the driven path far more closely. If nothing is accepted,
    the closest failing attempt is returned as a best effort.

    With **no reference at all** - neither a length nor a path - only the first rung is issued: every
    verdict would be uncomputable, so a later rung could only be preferred arbitrarily. Its
    ``validation["valid"]`` is then *unverified* rather than *checked and rejected*.
    """
    if routing_service != "osrm":
        raise ValueError(f"Unsupported routing_service: {routing_service!r}")

    start_proj = routing_task["start_proj"]
    goal_proj = routing_task["goal_proj"]
    map_name = routing_task["map_name"]
    epsg = routing_task["epsg"]
    start_heading = routing_task["start_heading"]
    goal_heading = routing_task.get("goal_heading")
    route_polyline = routing_task.get("route_polyline")
    nuplan_route_length_m = routing_task.get("nuplan_route_length_m")
    has_reference = isinstance(nuplan_route_length_m, (int, float)) or bool(route_polyline)
    path_check = PATH_CHECKS[routing_task.get("reference_kind", "trajectory")]

    attempts = route_attempts(
        start=routing_task["start"],
        start_proj=start_proj,
        goal=routing_task.get("goal", goal_proj),
        goal_proj=goal_proj,
        vias=routing_task.get("route_vias"),
    )

    # Shape of an empty answer; every real task has attempts, so the walk below always replaces it.
    selection = RouteSelection(
        routing_output=None,
        description=None,
        maneuver_positions=[],
        validation={
            "valid": True,
            "nuplan_length_m": (
                float(nuplan_route_length_m)
                if isinstance(nuplan_route_length_m, (int, float))
                else None
            ),
            "osrm_length_m": None,
            "ratio_nuplan_over_osrm": None,
            "relative_tolerance": ROUTE_LENGTH_REL_TOLERANCE,
        },
        strategy=None,
        start=start_proj,
        goal=goal_proj,
    )

    # Closest-to-the-driven-path attempt among those that did not pass, as (deviation, selection).
    narrow_fallback: Optional[tuple] = None
    # Best length-agreeing attempt seen so far, as (deviation, selection).
    best_valid: Optional[tuple] = None
    for attempt in attempts:
        routing_output, lg_description, maneuver_positions, route_validation = (
            execute_and_validate_routing_request(
                attempt.start,
                attempt.goal,
                map_name,
                epsg,
                routing_service,
                start_heading,
                goal_heading,
                attempt.vias,
                nuplan_route_length_m,
                bearing_tol=attempt.bearing_tol,
                route_polyline=route_polyline,
                path_check=path_check,
            )
        )
        candidate = RouteSelection(
            routing_output=routing_output,
            description=lg_description,
            maneuver_positions=maneuver_positions,
            validation=route_validation,
            strategy=attempt.strategy,
            start=attempt.start,
            goal=attempt.goal,
        )
        selection = candidate
        if not has_reference:
            return candidate
        if route_validation["valid"]:
            deviation = route_validation.get("path_deviation_p95_m")
            # Without a deviation there is nothing to compare on, so the first accepted
            # attempt wins, exactly as before the geometry check existed.
            if deviation is None:
                best_valid = (float("-inf"), candidate)
                break
            if best_valid is None or deviation < best_valid[0]:
                best_valid = (deviation, candidate)
            if deviation <= PATH_DEVIATION_ACCEPT_P95_M:
                break
        else:
            # Nothing has validated yet. Remember the attempt that came closest to the driven
            # path, so a run that fails outright still reports its best effort - useful when
            # inspecting why a scenario was rejected. Attempts with no measurable geometry all
            # tie, and the last of them wins, which is the previous behaviour.
            deviation = route_validation.get("path_deviation_mean_m")
            key = deviation if deviation is not None else float("inf")
            if narrow_fallback is None or key <= narrow_fallback[0]:
                narrow_fallback = (key, candidate)

    if best_valid is not None:
        return best_valid[1]
    if narrow_fallback is not None:
        return narrow_fallback[1]
    return selection


def process_routing_task(routing_task: Dict, routing_service: str) -> Dict:
    map_name = routing_task["map_name"]
    epsg = routing_task["epsg"]
    routing_horizon_s = routing_task["routing_horizon_s"]
    start_proj = routing_task["start_proj"]
    start_heading = routing_task["start_heading"]
    route_roadblock_ids = routing_task.get("route_roadblock_ids")
    route_connectivity_gaps = routing_task.get("route_connectivity_gaps") or []

    selection = select_route(routing_task, routing_service)
    routing_output = selection.routing_output
    lg_description = selection.description
    maneuver_positions = selection.maneuver_positions
    route_validation = selection.validation
    routing_strategy = selection.strategy
    route_start = selection.start
    route_goal = selection.goal

    result: Dict = {
        "token": routing_task["token"],
        "scenario_data": {
            "map_name": map_name,
            "scenario_type": routing_task["scenario_type"],
        },
        "routing_data": {
            "route_start": list(route_start),
            "route_end": list(route_goal),
            "route_epsg": int(epsg),
            "routing_horizon_s": float(routing_horizon_s),
            "route_roadblock_ids": route_roadblock_ids,
            "route_connectivity_gaps": route_connectivity_gaps,
            "route_description": lg_description,
            "route_maneuver_positions": maneuver_positions,
            "route_directions": routing_output,
            "valid_route": bool(route_validation["valid"]),
            "route_validation": route_validation,
            "routing_strategy": routing_strategy,
        },
    }

    interplan_goals = routing_task.get("interplan_goals")
    if interplan_goals is not None:
        routing_data_interplan = {}
        for variant in ("left", "right", "straight"):
            variant_goal_abs = interplan_goals.get(variant)
            if variant_goal_abs is None:
                (
                    variant_goal,
                    variant_output,
                    variant_description,
                    variant_maneuver_pos,
                ) = (None, None, None, None)
            else:
                variant_output, variant_description, variant_maneuver_pos = execute_routing_request(
                    start_proj,
                    variant_goal_abs,
                    map_name,
                    epsg,
                    routing_service=routing_service,
                    start_heading=start_heading,
                )
                variant_goal = list(variant_goal_abs)

            routing_data_interplan[f"route_end_{variant}"] = variant_goal
            routing_data_interplan[f"route_description_{variant}"] = variant_description
            routing_data_interplan[f"route_directions_{variant}"] = variant_output
            routing_data_interplan[f"route_maneuver_positions_{variant}"] = variant_maneuver_pos

        result["routing_data_interplan"] = routing_data_interplan

    return result


# Per-worker copy of the settings that are identical for every scenario. Installed once by
# ``init_scenario_worker`` instead of being pickled alongside each submitted scenario, which
# matters for ``interplan_goals_by_token`` (thousands of entries, otherwise re-sent per task).
_WORKER_SETTINGS: Dict = {}


def init_scenario_worker(
    route_goal_horizon_s: int,
    sample_frequency_hz: int,
    interplan_goals_by_token: Optional[Dict],
    routing_service: str,
) -> None:
    """``ProcessPoolExecutor`` initializer: pin the run-wide settings into this worker."""
    _WORKER_SETTINGS.update(
        route_goal_horizon_s=route_goal_horizon_s,
        sample_frequency_hz=sample_frequency_hz,
        interplan_goals_by_token=interplan_goals_by_token,
        routing_service=routing_service,
    )


def build_and_route_scenario(scenario) -> Dict:
    """Derive the routing task for ``scenario`` and route it, in the worker.

    Both halves run here so they overlap across workers. Building a task is the far more expensive
    half (map queries over the driven trajectory); when it ran in the parent process it was
    single-threaded work on the critical path while every worker sat idle waiting for it.
    """
    routing_task = build_routing_task_from_scenario(
        scenario,
        route_goal_horizon_s=_WORKER_SETTINGS["route_goal_horizon_s"],
        sample_frequency_hz=_WORKER_SETTINGS["sample_frequency_hz"],
        interplan_goals_by_token=_WORKER_SETTINGS["interplan_goals_by_token"],
    )
    return process_routing_task(routing_task, _WORKER_SETTINGS["routing_service"])


def build_and_route_alternative(item) -> Dict:
    """Worker entry point for an alternative-route dataset row.

    Mirrors :func:`build_and_route_scenario`, but the route is handed in rather than discovered:
    ``item`` is ``(scenario, alternative)`` where ``alternative`` carries at least
    ``route_roadblock_ids``, ``goal_position`` and the token to write the row under.

    Building and routing both run in the worker for the same reason as the scenario path -- the
    map queries behind the centerline and the vias are the expensive half, and leaving them in the
    parent would serialise them against otherwise-idle workers.
    """
    scenario, alternative = item
    routing_task = build_routing_task_from_alternative_route(
        scenario,
        alternative["route_roadblock_ids"],
        alternative["goal_position"],
        token=alternative.get("alt_token"),
        route_goal_horizon_s=_WORKER_SETTINGS["route_goal_horizon_s"],
    )
    row = process_routing_task(routing_task, _WORKER_SETTINGS["routing_service"])

    # Provenance so a row can be traced back to the scenario and the alternative it describes,
    # which the scenario-derived rows do not need because their token already says it.
    row["routing_data"]["source_token"] = alternative.get("source_token")
    row["routing_data"]["alt_index"] = alternative.get("alt_index")
    row["routing_data"]["instruction"] = alternative.get("instruction")
    return row
