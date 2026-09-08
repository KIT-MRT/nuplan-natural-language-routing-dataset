import math
from typing import Dict, Optional, Sequence, Tuple

import requests
from pyproj import Transformer as ProjTransformer


def _pad_to_fixed_length(positions, target_len: int, pad_value=None):
    """Pad or truncate to exactly ``target_len`` entries.

    Downstream consumers turn these into tensors, so the length must not vary with the number of
    maneuvers a route happens to have.
    """
    if pad_value is None:
        pad_value = [0, 0]
    padded = list(positions)
    # Build each pad entry separately: repeating a list would share one object across
    # every padded slot, so mutating one would mutate them all.
    padded.extend(
        list(pad_value) if isinstance(pad_value, list) else pad_value
        for _ in range(target_len - len(padded))
    )
    return padded[:target_len]


def request_route(
    start,
    destination,
    server_url: Optional[str] = "https://router.project-osrm.org",
    start_bearing: Optional[float] = None,
    goal_bearing: Optional[float] = None,
    bearing_tol: float = 45.0,
    vias: Optional[Sequence[Tuple[float, float]]] = None,
):
    """
    Get driving directions between start and destination using OSRM.
    Args:
        start (tuple): A tuple of (latitude, longitude) for the starting point.
        destination (tuple): A tuple of (latitude, longitude) for the destination point.
        server_url (str, optional): URL of the OSRM server. Defaults to the public OSRM server.
        start_bearing (float, optional): True-north compass bearing (degrees, clockwise from
            north) that the departure at ``start`` must fall within. Constrains OSRM's snap so it
            cannot land on the oncoming carriageway. ``None`` leaves the departure unconstrained.
        goal_bearing (float, optional): True-north compass bearing (degrees, clockwise from
            north) that the arrival at ``destination`` must fall within. Same purpose as
            ``start_bearing`` but for the last waypoint. ``None`` leaves it unconstrained.
        bearing_tol (float): Allowed deviation (degrees) around ``start_bearing``/``goal_bearing``.
        vias (sequence of (lat, lon), optional): Intermediate waypoints the route must pass
            through, in order, between ``start`` and ``destination``. OSRM visits them all, so the
            returned route follows the intended branch instead of the global shortest path. Each
            extra waypoint adds one leg to the response. ``None`` reproduces the original
            two-point request.
    Returns:
        dict: The routing information returned by OSRM.
    """
    # coords in OSRM order: lon,lat;lon,lat;... over [start] + vias + [destination].
    coords = [start, *(vias or []), destination]
    coord_str = ";".join(f"{c[1]},{c[0]}" for c in coords)
    url = f"{server_url}/route/v1/driving/{coord_str}?overview=full&steps=true"
    # A via's third element, when present, is its travel bearing.
    via_bearings = [(v[2] if len(v) > 2 else None) for v in (vias or [])]
    if (
        start_bearing is not None
        or goal_bearing is not None
        or any(b is not None for b in via_bearings)
    ):
        # OSRM requires either zero bearings or exactly one per waypoint. Vias carry their own
        # bearing where known; anything unknown stays unconstrained ("").
        tol = int(round(bearing_tol))
        bearing_entries = ["" for _ in coords]
        if start_bearing is not None:
            bearing_entries[0] = f"{int(round(start_bearing)) % 360},{tol}"
        if goal_bearing is not None:
            bearing_entries[-1] = f"{int(round(goal_bearing)) % 360},{tol}"
        for offset, via_bearing in enumerate(via_bearings):
            if via_bearing is not None:
                bearing_entries[offset + 1] = f"{int(round(via_bearing)) % 360},{tol}"
        url += "&bearings=" + ";".join(bearing_entries)
    if vias:
        # Mark ONLY the start (index 0) and destination (last index) as real waypoints. The vias
        # become "through" points: OSRM must route through them (so the intended branch is forced)
        # but does NOT split the route at them. This avoids per-via arrive/depart artifacts entirely.
        url += f"&waypoints=0;{len(coords) - 1}"
    # request with timeout (more robust with retry loop in calling function)
    return requests.get(url, timeout=(2.0, 10.0)).json()


def route_to_instructions(osrm_response):
    routes = (osrm_response or {}).get("routes") or []
    if not routes:
        # OSRM answered something other than "Ok" (e.g. NoSegment when a waypoint's bearing
        # constraint can't be met). Return no description so the caller's fallback ladder can
        # move on to the next attempt instead of this raising.
        return ""
    routing_steps = routes[0]["legs"][0]["steps"]
    mod_routing_steps = []
    for step in routing_steps:
        # round distance
        distance = step["distance"]
        distance_round_to_10m = round(distance / 10.0) * 10
        # determine maneuver
        maneuver = step.get("maneuver")
        maneuver_type = maneuver.get("type", None)
        maneuver_modifier = maneuver.get("modifier", None)
        maneuver_exit = maneuver.get("exit", None)
        # Number of intersections before maneuver
        intersections = step.get("intersections")
        intersections_before_next_maneuver = max(len(intersections) - 1, 0)

        mod_routing_steps.append(
            {
                "type": maneuver_type,
                "modifier": maneuver_modifier,
                "exit": maneuver_exit,
                "distance_to_next_maneuver": distance_round_to_10m,
                "intersections_before_next_maneuver": intersections_before_next_maneuver,
            }
        )

    routing_description = []
    for i in range(len(mod_routing_steps)):
        step_description = maneuver_to_instruction(
            mod_routing_steps[i]["type"],
            mod_routing_steps[i]["modifier"],
            mod_routing_steps[i]["exit"],
        )
        if i > 0:
            distance = mod_routing_steps[i - 1]["distance_to_next_maneuver"]
            step_description = step_description + f" in {distance} meters"
        routing_description.append(step_description)

    return "; ".join(routing_description)


def maneuver_to_instruction(type, modifier=None, exit=None):
    """
    Convert a (type, modifier, exit) maneuver into a simple driving instruction.
    """

    # Normalize inputs
    t = type.lower().strip() if type else ""
    m = modifier.lower().strip() if modifier else None

    # Helper to format direction
    def dir_text(mod):
        return mod.replace("_", " ") if mod else ""

    # ---- Special maneuvers ----
    if t == "depart":
        if m:
            return f"Depart heading {dir_text(m)}"
        return "Depart"

    if t == "arrive":
        if m:
            return f"Arrive on the {dir_text(m)}"
        return "Arrive at destination"

    if t == "new name":
        if m:
            return f"Continue {dir_text(m)}"
        return "Continue"

    if t == "notification":
        if m:
            return f"Continue {dir_text(m)}"
        return "Continue"

    # ---- Ramps ----
    if t == "on ramp":
        return f"Take the on-ramp {dir_text(m)}" if m else "Take the on-ramp"

    if t == "off ramp":
        return f"Take the off-ramp {dir_text(m)}" if m else "Take the off-ramp"

    # ---- Merge / fork / lane ----
    if t == "merge":
        return f"Merge {dir_text(m)}" if m else "Merge"

    if t == "fork":
        return f"Keep {dir_text(m)} at the fork" if m else "Keep to the fork"

    if t == "use lane":
        return "Use the indicated lane"

    if t == "end of road":
        return f"Turn {dir_text(m)} at the end of the road" if m else "Road ends"

    # ---- Roundabouts ----
    if t in ("roundabout", "rotary"):
        if exit:
            return f"At the roundabout, take exit {exit}"
        return "Enter the roundabout"

    if t == "roundabout turn":
        return f"At the roundabout, turn {dir_text(m)}" if m else "At the roundabout, turn"

    # ---- Continue ----
    if t == "continue":
        return f"Continue {dir_text(m)}" if m else "Continue"

    # ---- Default / turn / unknown future types ----
    # Treat unknown types like "turn"
    if m:
        return f"Turn {dir_text(m)}"

    # Fallback if absolutely nothing usable
    return "Continue"


def extract_maneuver_positions(osrm_response, epsg: int, max_maneuvers: int = 20):
    maneuver_positions = []
    transformer = ProjTransformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    routes = (osrm_response or {}).get("routes") or []
    if not routes:
        # Non-"Ok" OSRM response (see route_to_instructions) — no maneuvers to report.
        return _pad_to_fixed_length(maneuver_positions, max_maneuvers)
    steps = routes[0]["legs"][0]["steps"]
    for step in steps:
        maneuver = step.get("maneuver", {})
        location = maneuver.get("location", None)
        if location:
            x, y = transformer.transform(location[0], location[1])
            maneuver_positions.append([x, y])
    return _pad_to_fixed_length(maneuver_positions, max_maneuvers)


def extract_maneuver_headings(osrm_response, max_maneuvers: int = 20):
    """Heading of travel *after* each maneuver, in map-frame radians.

    OSRM reports ``bearing_after`` as a compass bearing: degrees clockwise from north.
    Map/UTM coordinates are east-handed and counter-clockwise, so the conversion is
    ``theta = radians(90 - bearing)``.

    This is orientation the language cannot express. "Turn left" says how the heading
    changes, never what it becomes, so without it a maneuver is a bare point and the
    model cannot tell which of two lanes leaving that point continues along the route.

    Entries are index-aligned with :func:`extract_maneuver_positions` -- both iterate the
    same ``steps`` list and skip on the same condition -- so a single validity mask
    covers both. Padding is 0.0, which is indistinguishable from a real east-pointing
    heading; use the position mask, never a zero test, to find real entries.

    Note: bearings are relative to true north while the coordinates are UTM grid north.
    Grid convergence between them is under ~2 degrees for these maps and is not corrected.
    """
    headings = []
    routes = (osrm_response or {}).get("routes") or []
    if not routes:
        # Non-"Ok" OSRM response (see route_to_instructions) -- no maneuvers to report.
        return _pad_to_fixed_length(headings, max_maneuvers, pad_value=0.0)
    steps = routes[0]["legs"][0]["steps"]
    for step in steps:
        maneuver = step.get("maneuver", {})
        # Gate on `location`, exactly as extract_maneuver_positions does: a step without
        # one contributes no position, and appending a heading for it would shift every
        # later heading out of alignment with its position.
        if maneuver.get("location", None):
            bearing = maneuver.get("bearing_after", None)
            headings.append(
                math.radians(90.0 - float(bearing)) if bearing is not None else 0.0
            )
    return _pad_to_fixed_length(headings, max_maneuvers, pad_value=0.0)


def first_turn_distance_m(routing_output: Dict) -> Optional[float]:
    """Depart-leg distance (m) if OSRM's second step is an immediate left/right turn.

    OSRM reports this depart-to-turn distance as the distance to the middle of the
    intersection, whereas nuPlan's route length takes the direct line to the turn. Returns
    ``None`` when the second step isn't a left/right turn (or the response is malformed).
    """
    routes = (routing_output or {}).get("routes") or []
    if not routes or not isinstance(routes[0], dict):
        return None
    legs = routes[0].get("legs") or []
    if not legs:
        return None
    steps = legs[0].get("steps") or []
    if len(steps) < 2:
        return None
    second_maneuver = steps[1].get("maneuver") or {}
    if second_maneuver.get("type") != "turn":
        return None
    modifier = (second_maneuver.get("modifier") or "").lower()
    if "left" not in modifier and "right" not in modifier:
        return None
    distance = steps[0].get("distance")
    return float(distance) if isinstance(distance, (int, float)) and distance <= 50 else None
