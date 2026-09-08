import math
from typing import Any, List, Optional, Sequence, Tuple

from nuplan.common.maps.maps_datatypes import SemanticMapLayer

MIN_VIA_SPACING_M = 8.0
# Short roadblocks are excluded because their midpoints get snapped by OSRM onto the oncoming
# carriageway or a crossing street, which drags the route off the driven path.
MIN_ROADBLOCK_LENGTH_FOR_VIA_M = 30.0
# Cap on vias per OSRM request. A route is already fully determined by a few well-placed
# waypoints, and each additional via is another opportunity for OSRM to snap onto the wrong way,
# so coverage past this point costs reliability without buying route fidelity.
MAX_ROUTE_VIAS = 3
# (x, y, heading) in the map frame; heading is the lane tangent at the waypoint.
Waypoint = Optional[Tuple[float, float, float]]
# Helpers that only read x/y and pass entries through accept any waypoint width (internally a
# 4th element carries the roadblock length; see _via_candidates).
AnyWaypoint = Optional[Tuple[float, ...]]


def _angle_diff(a: float, b: float) -> float:
    """Smallest signed angle difference a-b in radians."""
    return math.atan2(math.sin(a - b), math.cos(a - b))


def _lane_midpoint_pose(lane: Any) -> Optional[Tuple[float, float, float]]:
    """Return (x, y, heading) at the lane baseline midpoint."""
    path = lane.baseline_path.discrete_path
    if not path:
        return None
    mid = path[len(path) // 2]
    return (float(mid.x), float(mid.y), float(mid.heading))


def _select_middle_lane(lanes) -> Optional[Any]:
    """Pick the lane whose midpoint is closest to the centroid of lane midpoints."""
    poses = []
    for lane in lanes:
        pose = _lane_midpoint_pose(lane)
        if pose is not None:
            poses.append((lane, pose))

    if not poses:
        return None
    if len(poses) == 1:
        return poses[0][0]

    cx = sum(p[1][0] for p in poses) / len(poses)
    cy = sum(p[1][1] for p in poses) / len(poses)
    return min(poses, key=lambda p: (p[1][0] - cx) ** 2 + (p[1][1] - cy) ** 2)[0]


def _lateral_foot_on_path(path, ego_x: float, ego_y: float) -> Optional[Tuple[float, float, float]]:
    """Project (ego_x, ego_y) laterally to a polyline path and return (x, y, heading)."""
    if not path:
        return None
    if len(path) == 1:
        p = path[0]
        return (float(p.x), float(p.y), float(p.heading))

    best_dist2 = float("inf")
    best = None
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        abx = float(b.x - a.x)
        aby = float(b.y - a.y)
        denom = abx * abx + aby * aby
        if denom <= 0.0:
            continue

        apx = float(ego_x - a.x)
        apy = float(ego_y - a.y)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        fx = float(a.x) + t * abx
        fy = float(a.y) + t * aby
        d2 = (fx - ego_x) ** 2 + (fy - ego_y) ** 2
        if d2 < best_dist2:
            best_dist2 = d2
            heading = math.atan2(aby, abx)
            best = (fx, fy, heading)

    return best


def _roadblock_middle_lane_of(map_api, lane) -> Optional[Any]:
    """Resolve ``lane``'s parent roadblock and return that roadblock's middle lane."""
    rid = lane.get_roadblock_id()
    if rid is None:
        return None

    roadblock = map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK)
    if roadblock is None:
        roadblock = map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK_CONNECTOR)
    if roadblock is None or not roadblock.interior_edges:
        return None

    return _select_middle_lane(roadblock.interior_edges)


def _roadblock_centerline_foot_from_lane(
    map_api, lane, ego_x: float, ego_y: float
) -> Optional[Tuple[float, float, float]]:
    """Project ego laterally to the selected parent roadblock's middle-lane centerline."""
    middle_lane = _roadblock_middle_lane_of(map_api, lane)
    if middle_lane is None:
        return None
    return _lateral_foot_on_path(middle_lane.baseline_path.discrete_path, ego_x, ego_y)


def _roadblock_centerline_midpoint_from_lane(map_api, lane) -> Optional[Tuple[float, float, float]]:
    """Snap to the selected parent roadblock's middle-lane longitudinal AND lateral center.

    Same anchor as the static via-points in ``roadblock_midpoints`` below, rather than a
    lateral-only foot that preserves the point's original along-track station.
    """
    middle_lane = _roadblock_middle_lane_of(map_api, lane)
    if middle_lane is None:
        return None
    return _lane_midpoint_pose(middle_lane)


def _project_point_to_route_roadblock_centerline(
    map_api,
    route_roadblock_ids,
    point,
    point_heading: float,
    radius_m: float,
    off_route_penalty_m: float,
    start_roadblock_id: Optional[str] = None,
) -> Tuple[List[float], float, Optional[str]]:
    """Project a point to best-matched lane, then to parent roadblock middle-lane centerline.

    If ``start_roadblock_id`` is given and the point resolves to a *different* roadblock,
    the anchor is also centered longitudinally (like the via midpoints in
    ``roadblock_midpoints``) instead of only laterally, for routing stability. When the point
    is on the start's own roadblock (or ``start_roadblock_id`` is not given), only the lateral
    foot is used so the anchor still reflects the point's actual along-track position.
    """
    route_ids = set(route_roadblock_ids or [])

    proximal = map_api.get_proximal_map_objects(
        point=point,
        radius=radius_m,
        layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
    )
    candidate_lanes = proximal.get(SemanticMapLayer.LANE, []) + proximal.get(
        SemanticMapLayer.LANE_CONNECTOR, []
    )

    best_lane = None
    best_cost = float("inf")
    for lane in candidate_lanes:
        path = lane.baseline_path.discrete_path
        if not path:
            continue

        nearest = min(path, key=lambda p: (p.x - point.x) ** 2 + (p.y - point.y) ** 2)
        if abs(_angle_diff(float(nearest.heading), point_heading)) > math.pi / 2:
            continue

        lateral_dist = math.sqrt((nearest.x - point.x) ** 2 + (nearest.y - point.y) ** 2)
        on_route_penalty = 0.0 if lane.get_roadblock_id() in route_ids else off_route_penalty_m
        cost = lateral_dist + on_route_penalty
        if cost < best_cost:
            best_cost = cost
            best_lane = lane

    if best_lane is None:
        return [float(point.x), float(point.y)], point_heading, None

    if start_roadblock_id is not None and best_lane.get_roadblock_id() != start_roadblock_id:
        anchor = _roadblock_centerline_midpoint_from_lane(map_api, best_lane)
    else:
        anchor = _roadblock_centerline_foot_from_lane(
            map_api,
            best_lane,
            float(point.x),
            float(point.y),
        )
    if anchor is None:
        fallback = _lane_midpoint_pose(best_lane)
        if fallback is None:
            return [float(point.x), float(point.y)], point_heading, None
        return [fallback[0], fallback[1]], fallback[2], best_lane.get_roadblock_id()

    return [anchor[0], anchor[1]], anchor[2], best_lane.get_roadblock_id()


def project_start_to_route_midpoint(
    map_api,
    route_roadblock_ids,
    ego_point,
    ego_heading: float,
    radius_m: float = 5.0,
    off_route_penalty_m: float = 5.0,
) -> Tuple[List[float], float, Optional[str]]:
    """Project ego to best-matched lane, then to lateral center of that roadblock's middle lane.

    Returns ``([x, y], heading, roadblock_id)`` where heading is the tangent at the projected foot.
    Falls back to ego pose when no suitable lane can be found. ``map_api``/``route_roadblock_ids``
    are taken directly (rather than a nuPlan ``scenario``) so this also serves live planning, where
    they come from ``PlannerInitialization``/route-correction instead of a logged scenario.
    """
    return _project_point_to_route_roadblock_centerline(
        map_api,
        route_roadblock_ids,
        ego_point,
        float(ego_heading),
        radius_m,
        off_route_penalty_m,
    )


def project_goal_to_route_centerline(
    map_api,
    route_roadblock_ids,
    goal_point,
    goal_heading: float,
    radius_m: float = 5.0,
    off_route_penalty_m: float = 5.0,
    start_roadblock_id: Optional[str] = None,
) -> Tuple[List[float], float, Optional[str]]:
    """Project goal to best-matched lane, then to that roadblock's middle-lane centerline.

    When ``start_roadblock_id`` is given and the goal resolves to a roadblock other than the
    start's, the goal is also centered longitudinally (not just laterally) for routing
    stability, matching the anchor used for intermediate via points.
    """
    return _project_point_to_route_roadblock_centerline(
        map_api,
        route_roadblock_ids,
        goal_point,
        float(goal_heading),
        radius_m,
        off_route_penalty_m,
        start_roadblock_id=start_roadblock_id,
    )


def truncate_waypoints_before_goal(
    waypoints: Sequence[AnyWaypoint],
    goal_route_index: Optional[int],
) -> Sequence[AnyWaypoint]:
    """Cut waypoints at the roadblock before goal's route index.

    Waypoint ``i`` corresponds to ``route_roadblock_ids[i]``. If the goal lies at route
    index ``k``, we keep ``waypoints[:k]`` (up to the roadblock before the goal roadblock).
    If ``goal_route_index`` cannot be resolved, return waypoints unchanged (fallback).
    """
    if goal_route_index is None:
        return waypoints
    if goal_route_index <= 0:
        return []
    return waypoints[:goal_route_index]


def resolve_goal_route_index(
    map_api,
    route_roadblock_ids,
    goal_point,
    radius_m: float = 2.0,
    prefer_furthest: bool = True,
) -> Optional[int]:
    """Resolve a point's roadblock index along route_roadblock_ids.

    Fallback behavior: if proximity query misses, use nearest roadblock/connector map objects.
    ``prefer_furthest`` controls the tie-break when several candidates overlap the query point:
    True (goal resolution, the default) biases toward the furthest-along match. Current-position
    resolution uses False instead (see ``resolve_current_route_index``), since biasing the ego's
    own position toward "furthest along" would assume it has already passed roadblocks it hasn't
    actually reached, undercounting any route length measured from that point.
    """
    if goal_point is None or not route_roadblock_ids:
        return None

    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    route_pos = {rid: idx for idx, rid in enumerate(route_roadblock_ids)}

    proximal = map_api.get_proximal_map_objects(
        point=goal_point,
        radius=radius_m,
        layers=layers,
    )
    candidates = proximal.get(SemanticMapLayer.ROADBLOCK, []) + proximal.get(
        SemanticMapLayer.ROADBLOCK_CONNECTOR, []
    )
    indices = [route_pos[obj.id] for obj in candidates if obj.id in route_pos]
    if indices:
        return max(indices) if prefer_furthest else min(indices)

    for layer in layers:
        nearest_id, _ = map_api.get_distance_to_nearest_map_object(
            point=goal_point,
            layer=layer,
        )
        if nearest_id in route_pos:
            return route_pos[nearest_id]

    return None


def resolve_current_route_index(
    map_api,
    route_roadblock_ids,
    current_point,
    radius_m: float = 2.0,
) -> Optional[int]:
    """Resolve ego/current roadblock index along route_roadblock_ids.

    Uses the same robust proximity + nearest fallback logic as goal lookup, but prefers the
    earliest overlapping candidate (``prefer_furthest=False``) so the ego is never resolved to a
    roadblock further along the route than it has actually reached.
    """
    return resolve_goal_route_index(
        map_api,
        route_roadblock_ids,
        current_point,
        radius_m=radius_m,
        prefer_furthest=False,
    )


def _is_intersection_proximal(roadblock) -> bool:
    """Heuristic: nodes with multi-in/out connections are intersection-proximal."""
    incoming = getattr(roadblock, "incoming_edges", []) or []
    outgoing = getattr(roadblock, "outgoing_edges", []) or []
    return len(incoming) > 2 or len(outgoing) > 2


def _via_candidates(
    map_api,
    route_roadblock_ids,
    min_roadblock_length_m: float = MIN_ROADBLOCK_LENGTH_FOR_VIA_M,
    drop_intersection_proximal: bool = False,
) -> Optional[List[Optional[Tuple[float, float, float, float]]]]:
    """Like ``roadblock_midpoints`` but each entry also carries its lane length.

    The length is what lets ``select_spread_vias`` prefer long, mid-block roadblocks, where OSRM's
    snap is unambiguous, over short ones.
    """
    if not route_roadblock_ids:
        return None

    midpoints: List[Optional[Tuple[float, float, float, float]]] = []
    for rid in route_roadblock_ids[:-1]:
        # Resolve connectors too, not just roadblocks: a driven route is connector-rich (a
        # turnaround loop is made almost entirely of short connectors), and resolving only
        # ROADBLOCK silently yielded no via for every one of them.
        roadblock = map_api.get_map_object(
            rid, SemanticMapLayer.ROADBLOCK
        ) or map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if roadblock is None or not roadblock.interior_edges:
            midpoints.append(None)
            continue

        if drop_intersection_proximal and _is_intersection_proximal(roadblock):
            midpoints.append(None)
            continue

        lane = _select_middle_lane(roadblock.interior_edges)
        if lane is None:
            midpoints.append(None)
            continue
        if lane.baseline_path.length < float(min_roadblock_length_m):
            midpoints.append(None)
            continue

        path = lane.baseline_path.discrete_path
        mid = path[len(path) // 2]
        # Carry the lane heading, not just the position: sent to OSRM as a per-via bearing it
        # pins the snap to a way travelling this direction, which is what stops short roadblocks'
        # midpoints being matched onto the oncoming carriageway or a crossing street.
        midpoints.append((mid.x, mid.y, mid.heading, float(lane.baseline_path.length)))

    if all(p is None for p in midpoints):
        return None
    return midpoints


def roadblock_midpoints(
    map_api,
    route_roadblock_ids,
    min_roadblock_length_m: float = MIN_ROADBLOCK_LENGTH_FOR_VIA_M,
    drop_intersection_proximal: bool = False,
) -> Optional[List[Waypoint]]:
    """Derive one ``(x, y, heading)`` midpoint waypoint per route roadblock (excluding the last).

    The heading is the lane tangent at the midpoint, carried so callers can constrain the
    direction OSRM snaps the via to (see ``build_route_vias``).
    """
    candidates = _via_candidates(
        map_api,
        route_roadblock_ids,
        min_roadblock_length_m=min_roadblock_length_m,
        drop_intersection_proximal=drop_intersection_proximal,
    )
    if candidates is None:
        return None
    return [None if c is None else (c[0], c[1], c[2]) for c in candidates]


def select_spread_vias(
    candidates: Sequence[Tuple[float, ...]],
    max_vias: int = MAX_ROUTE_VIAS,
) -> Optional[List[Tuple[float, float, float]]]:
    """Reduce via candidates to at most ``max_vias``, spread along the route.

    A route is already pinned by a handful of well-placed vias — one per stretch is enough to fix
    which branch was taken — while every extra via is another chance for OSRM to snap onto the
    wrong way. So the candidates are split into ``max_vias`` consecutive buckets and the **longest**
    roadblock in each is used, keeping vias spread out and on the segments where the snap is least
    ambiguous. Route order is preserved.
    """
    if not candidates:
        return None
    if len(candidates) <= max_vias:
        return [(c[0], c[1], c[2]) for c in candidates]

    total = len(candidates)
    chosen: List[Tuple[float, float, float]] = []
    for bucket in range(max_vias):
        lo = (bucket * total) // max_vias
        hi = ((bucket + 1) * total) // max_vias
        window = candidates[lo:hi] or candidates[lo : lo + 1]
        best = max(window, key=lambda c: c[3])
        chosen.append((best[0], best[1], best[2]))
    return chosen


def advance_via_index(prev_k: int, current_roadblock_id, route_roadblock_ids) -> int:
    """Advance monotone cursor to ego's current roadblock index when available."""
    if (
        current_roadblock_id is not None
        and route_roadblock_ids
        and current_roadblock_id in route_roadblock_ids
    ):
        k = route_roadblock_ids.index(current_roadblock_id)
        if k > prev_k:
            return k
    return prev_k


def filter_ahead_vias(
    waypoints: Optional[Sequence[AnyWaypoint]],
    k: int,
    min_spacing_m: float = MIN_VIA_SPACING_M,
) -> Optional[List[Tuple[float, ...]]]:
    """Keep waypoints strictly ahead of k and enforce minimum spacing."""
    if not waypoints:
        return None

    kept: List[Tuple[float, ...]] = []
    for v in waypoints[k + 1 :]:
        if v is None:
            continue
        if not kept or math.hypot(v[0] - kept[-1][0], v[1] - kept[-1][1]) >= min_spacing_m:
            kept.append(v)

    return kept or None


def build_route_vias(
    map_api,
    route_roadblock_ids,
    ego_point,
    goal_point=None,
    min_spacing_m: float = MIN_VIA_SPACING_M,
    min_roadblock_length_m: float = MIN_ROADBLOCK_LENGTH_FOR_VIA_M,
    drop_intersection_proximal: bool = True,
    max_vias: int = MAX_ROUTE_VIAS,
) -> Optional[List[Tuple[float, float, float]]]:
    """Build a small set of static OSRM vias from route roadblocks.

    Takes ``map_api``/``route_roadblock_ids``/``ego_point`` directly (rather than a nuPlan
    ``scenario``), matching ``project_start_to_route_midpoint`` et al., so this also
    serves callers with an already-resolved route instead of only the scenario's raw stored route.

    Deliberately returns **few** vias (``max_vias``), and never one inside a junction if avoidable:
    a route is fully pinned by a handful of well-placed waypoints, whereas an intersection-proximal
    via is the case most likely to be snapped onto a crossing street or the oncoming carriageway.
    Junction roadblocks are therefore excluded first, and only reconsidered when that leaves no
    candidate at all — better a slightly risky via than an entirely unconstrained request.
    """
    if not route_roadblock_ids or route_roadblock_ids == [""]:
        return None

    candidates = _via_candidates(
        map_api,
        route_roadblock_ids,
        min_roadblock_length_m=min_roadblock_length_m,
        drop_intersection_proximal=drop_intersection_proximal,
    )
    if candidates is None and drop_intersection_proximal:
        # Junction-only route (e.g. a turnaround loop): fall back to allowing them.
        candidates = _via_candidates(
            map_api,
            route_roadblock_ids,
            min_roadblock_length_m=min_roadblock_length_m,
            drop_intersection_proximal=False,
        )
    if candidates is None:
        return None

    goal_route_index = resolve_goal_route_index(
        map_api,
        route_roadblock_ids,
        goal_point,
    )
    current_route_index = resolve_current_route_index(
        map_api,
        route_roadblock_ids,
        ego_point,
    )
    truncated = truncate_waypoints_before_goal(candidates, goal_route_index)

    k = current_route_index if current_route_index is not None else -1
    ahead = filter_ahead_vias(truncated, k=k, min_spacing_m=min_spacing_m)
    if not ahead:
        return None
    return select_spread_vias(ahead, max_vias=max_vias)
