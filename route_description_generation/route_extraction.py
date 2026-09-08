"""Deriving a routing task from the route the ego actually drove.

The route is read off the ground-truth trajectory rather than inferred by a graph search toward the
goal, so the branch taken at every fork is correct by construction. The raw devkit output needs
repairing in a few places; :func:`build_route_roadblock_ids_from_trajectory` documents which.
"""

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.utils import get_roadblock_ids_from_trajectory

from route_description_generation.geometry import polyline_length_m, trim_polyline
from route_description_generation.scenario_loading import MAP_EPSG
from route_description_generation.waypoints import (
    build_route_vias,
    project_goal_to_route_centerline,
    project_start_to_route_midpoint,
)

# How far the gap-bridging search may look when two consecutive driven roadblocks are not directly
# connected (see _bridge_roadblocks). Gaps come from the devkit skipping a roadblock in an
# ambiguous/overlapping area, so they span very few hops; this is generous but still bounded.
MAX_BRIDGE_DEPTH = 6

# How many later trajectory roadblocks to test when the next one is unreachable, before concluding
# the break is a real map discontinuity rather than a stray parallel connector at a junction.
REACHABLE_LOOKAHEAD = 5

# How much of the route already built may be discarded when it turns out the stray connector is
# the one already committed (so nothing following it can connect).
MAX_BACKTRACK = 3

# Minimum chord (metres) used to read a direction of travel off a trajectory. Well above the
# sub-millimetre jitter a stationary ego logs, well below the scale over which a road curves.
MIN_HEADING_SPAN_M = 1.0


def _get_roadblock(map_api, rb_id: str):
    """Resolve a roadblock or roadblock-connector by id."""
    return map_api.get_map_object(rb_id, SemanticMapLayer.ROADBLOCK) or map_api.get_map_object(
        rb_id, SemanticMapLayer.ROADBLOCK_CONNECTOR
    )


def _outgoing_ids(map_api, rb_id: str) -> Set[str]:
    roadblock = _get_roadblock(map_api, rb_id)
    if roadblock is None:
        return set()
    return {edge.id for edge in (getattr(roadblock, "outgoing_edges", None) or [])}


def _bridge_roadblocks(
    map_api,
    from_id: str,
    to_id: str,
    max_depth: int = MAX_BRIDGE_DEPTH,
) -> Optional[List[str]]:
    """Shortest chain of roadblock ids strictly *between* ``from_id`` and ``to_id``.

    Returns ``[]`` when the two are already directly connected, or ``None`` when no chain exists
    within ``max_depth`` (a genuine map gap). Both endpoints are fixed, so unlike a goal-directed
    search there is no branch being chosen here — the result is simply the missing links.
    """
    if to_id in _outgoing_ids(map_api, from_id):
        return []

    # BFS forward from ``from_id``, remembering each node's predecessor to rebuild the chain.
    parents: Dict[str, str] = {from_id: ""}
    frontier = [from_id]
    for _ in range(max_depth):
        next_frontier: List[str] = []
        for rb_id in frontier:
            for nxt_id in _outgoing_ids(map_api, rb_id):
                if nxt_id in parents:
                    continue
                parents[nxt_id] = rb_id
                if nxt_id == to_id:
                    chain: List[str] = []
                    cursor = parents[nxt_id]
                    while cursor and cursor != from_id:
                        chain.append(cursor)
                        cursor = parents[cursor]
                    return list(reversed(chain))
                next_frontier.append(nxt_id)
        frontier = next_frontier
    return None


def build_route_roadblock_ids_from_trajectory(map_api, ego_states) -> List[str]:
    """Connected roadblock ids the ego actually drove through, in order.

    The route is *read off the ground-truth trajectory* rather than inferred by a graph search
    toward the goal, so the branch taken at every fork is correct by construction and the goal is
    guaranteed to lie on the route (see ``_append_goal_roadblock``).

    Three things can break the chain, and only the first two are repairable:

    - **Skipped roadblock.** ``get_roadblock_ids_from_trajectory`` appends a roadblock only once
      exactly one candidate contains the current point, so wherever that ambiguity never resolves
      an intermediate roadblock is missing. ``_bridge_roadblocks`` reinserts it.
    - **Spurious parallel connector.** Junctions contain several overlapping turn connectors, so
      as the ego crosses one, ``contains_point`` can flip between mutually unreachable
      alternatives only a few metres apart (e.g. ``19458 -> 19460 -> 19461 -> 19457``, none
      reachable from another). An id that cannot be reached from the route so far, when a *later*
      one can be, is one of these alternatives and is dropped in favour of the branch that keeps
      the route connected. The mirror case — where the stray is the id *already committed*, so
      nothing after it connects — is repaired by dropping back up to ``MAX_BACKTRACK`` ids, but
      never the whole prefix: a leading stray and a genuine break at the route's start are
      structurally identical (both have no outgoing edges), and discarding real driven roadblocks
      is the worse error, so a break in the opening stretch is reported rather than "fixed".
    - **Missing map edge.** Some nuPlan roadblocks have no ``outgoing_edges`` at all even though
      the ego demonstrably drives out of them (e.g. ``49253 -> 48808`` in us-ma-boston, where the
      predecessor has no outgoing and the successor no incoming edges). No search can bridge that;
      the pair is kept as-is, since the ego really did drive it, and reported by
      :func:`route_connectivity_gaps` so it is visible rather than silent.
    """
    states = list(ego_states)
    raw_ids = [rb_id for rb_id in get_roadblock_ids_from_trajectory(map_api, states) if rb_id]
    if not raw_ids:
        return []

    route_ids: List[str] = [raw_ids[0]]
    index = 1
    while index < len(raw_ids):
        rb_id = raw_ids[index]
        if rb_id == route_ids[-1]:
            index += 1
            continue

        bridge = _bridge_roadblocks(map_api, route_ids[-1], rb_id)
        if bridge is not None:
            route_ids.extend(bridge)
            route_ids.append(rb_id)
            index += 1
            continue

        # Unreachable from the route so far. If a nearby later id *is* reachable, everything up to
        # it was a parallel alternative the ego's points strayed onto — skip to that one instead.
        reconnect = None
        for ahead in range(index + 1, min(index + 1 + REACHABLE_LOOKAHEAD, len(raw_ids))):
            if raw_ids[ahead] == route_ids[-1]:
                continue
            ahead_bridge = _bridge_roadblocks(map_api, route_ids[-1], raw_ids[ahead])
            if ahead_bridge is not None:
                reconnect = (ahead, ahead_bridge)
                break
        if reconnect is not None:
            ahead, ahead_bridge = reconnect
            route_ids.extend(ahead_bridge)
            route_ids.append(raw_ids[ahead])
            index = ahead + 1
            continue

        # Mirror case: the stray is the id already committed, so nothing after it can connect.
        # Drop the tail back to the last id that does reach here (possibly all of it, when the
        # very first observations were strays).
        backtracked = False
        for drop in range(1, min(MAX_BACKTRACK, len(route_ids) - 1) + 1):
            kept = route_ids[:-drop]
            back_bridge = _bridge_roadblocks(map_api, kept[-1], rb_id)
            if back_bridge is not None:
                route_ids = kept + back_bridge + [rb_id]
                index += 1
                backtracked = True
                break
        if backtracked:
            continue

        # Nothing ahead or behind reconnects: a genuine discontinuity in the map.
        route_ids.append(rb_id)
        index += 1

    if states:
        route_ids = _replace_unreachable_start_roadblock(
            map_api,
            route_ids,
            states[0].rear_axle.point,
            float(states[0].rear_axle.heading),
        )
    return route_ids


def _replace_unreachable_start_roadblock(
    map_api, route_ids: List[str], start_point, start_heading: Optional[float] = None
) -> List[str]:
    """Swap a leading roadblock that cannot reach the rest of the route for one that can.

    The mirror of :func:`_append_goal_roadblock` at the other end. Where a lane forks just before
    an intersection, the ego waiting at the stop line sits inside *both* the through connector and
    the turn connector beside it, and the devkit can commit the turn - which no part of the driven
    route is reachable from. Every downstream step then inherits it: the start is projected onto
    the turning lane, and OSRM is asked to depart into a turn the ego never took.

    Only fires when the opening pair is already disconnected, so a route that starts cleanly is
    untouched; a genuine map break at the route's start stays reported, since no candidate
    containing the ego will reach the route either.
    """
    if len(route_ids) < 2 or route_ids[1] in _outgoing_ids(map_api, route_ids[0]):
        return route_ids

    candidates = []
    for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
        proximal = map_api.get_proximal_map_objects(start_point, 1.0, [layer])[layer]
        candidates.extend(
            obj for obj in proximal if obj.id != route_ids[0] and obj.contains_point(start_point)
        )
    if not candidates:
        return route_ids

    def deviation(candidate) -> float:
        if start_heading is None:
            return 0.0
        heading = _roadblock_heading_at(candidate, start_point)
        if heading is None:
            return math.pi
        return abs(math.atan2(math.sin(heading - start_heading), math.cos(heading - start_heading)))

    for candidate in sorted(candidates, key=deviation):
        if candidate.id in route_ids:
            continue
        bridge = _bridge_roadblocks(map_api, candidate.id, route_ids[1])
        if bridge is not None:
            return [candidate.id] + bridge + route_ids[1:]
    return route_ids


def route_connectivity_gaps(map_api, route_roadblock_ids) -> List[List[str]]:
    """Consecutive route pairs with no connecting edge in the map graph.

    After bridging these are only ever *missing map edges* (see
    ``build_route_roadblock_ids_from_trajectory``), i.e. a defect in the map rather than in the
    route. Persisted with the dataset so such scenarios can be found instead of silently looking
    like a malformed route.
    """
    return [
        [a, b]
        for a, b in zip(route_roadblock_ids, route_roadblock_ids[1:])
        if b not in _outgoing_ids(map_api, a)
    ]


def _roadblock_heading_at(roadblock, point) -> Optional[float]:
    """Travel heading of ``roadblock`` at the baseline point nearest ``point``.

    Sampled at the nearest point rather than at the lane midpoint because junction connectors
    curve: which way the roadblock "runs" is only meaningful locally.
    """
    best_heading = None
    best_distance = float("inf")
    for lane in roadblock.interior_edges:
        for pose in lane.baseline_path.discrete_path:
            distance = math.hypot(pose.x - point.x, pose.y - point.y)
            if distance < best_distance:
                best_distance = distance
                best_heading = float(pose.heading)
    return best_heading


def _append_goal_roadblock(
    map_api, route_ids: List[str], goal_point, goal_heading: Optional[float] = None
) -> List[str]:
    """Ensure the roadblock containing ``goal_point`` is the final route id.

    Uses ``contains_point`` — the same predicate ``get_roadblock_ids_from_trajectory`` uses — so
    this cannot disagree with the route the way a separate proximity/heading resolver would.

    When the goal falls inside an intersection, *several* roadblocks contain it: every crossing
    connector overlaps there. Picking whichever the map query returned first lands on one running
    across the ego's direction of travel and unreachable from the route, which then drags
    ``project_goal_to_route_centerline`` onto that crossing street and makes OSRM
    approach the goal from the wrong road entirely. Candidates are therefore ranked by whether the
    route can actually reach them and by how closely they run along ``goal_heading``, with the
    unranked first-hit order kept only as a last resort so behaviour is unchanged where the goal is
    unambiguous.
    """
    if route_ids:
        last = _get_roadblock(map_api, route_ids[-1])
        if last is not None and last.contains_point(goal_point):
            return route_ids

    holders = []
    for layer in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR):
        proximal = map_api.get_proximal_map_objects(goal_point, 1.0, [layer])[layer]
        holders.extend(obj for obj in proximal if obj.contains_point(goal_point))
    if not holders:
        return route_ids

    if len(holders) > 1:
        tail_id = route_ids[-1] if route_ids else None

        def rank(holder):
            # Reachable-from-the-route first: a roadblock the ego cannot drive into from where it
            # is cannot be the one it ends up in, whatever its orientation.
            reachable = tail_id is not None and (
                holder.id in route_ids
                or _bridge_roadblocks(map_api, tail_id, holder.id) is not None
            )
            deviation = math.pi
            if goal_heading is not None:
                heading = _roadblock_heading_at(holder, goal_point)
                if heading is not None:
                    deviation = abs(
                        math.atan2(
                            math.sin(heading - goal_heading),
                            math.cos(heading - goal_heading),
                        )
                    )
            return (not reachable, deviation)

        holders = sorted(holders, key=rank)

    # Prefer a holder already on the route (truncate to it); otherwise append + bridge.
    for holder in holders:
        if holder.id in route_ids:
            return route_ids[: route_ids.index(holder.id) + 1]

    goal_id = holders[0].id
    if not route_ids:
        return [goal_id]
    bridge = _bridge_roadblocks(map_api, route_ids[-1], goal_id)
    if bridge:
        route_ids.extend(bridge)
    route_ids.append(goal_id)
    return route_ids


def trajectory_arc_length_m(ego_states: Sequence) -> float:
    """Distance actually driven along ``ego_states``.

    Replaces walking route lanes and matching the goal to the nearest one: that matching had no
    heading check, so on a divided road it locked onto the *outbound* carriageway a few metres away
    and silently discarded the whole return leg of any route that doubles back.
    """
    return polyline_length_m(ego_state_points(ego_states))


def ego_state_points(ego_states: Sequence) -> List[Tuple[float, float]]:
    """Rear-axle ``(x, y)`` of each state, the polyline form the route checks compare against."""
    return [(float(s.rear_axle.point.x), float(s.rear_axle.point.y)) for s in ego_states]


def route_lane_chain(map_api, route_roadblock_ids, start_xy) -> List:
    """The connected chain of lanes a route is driven along, one lane per roadblock.

    Each lane is chosen to follow on from the previous one rather than by taking each roadblock's
    middle lane independently: A's middle lane need not feed B's, so concatenating them jogs a lane
    width at every join. Measured over val14 the connected chain tracks the driven route to within
    0.1% on length (median ratio 1.001), where naive concatenation drifts 3%.

    Returned as lane objects rather than points so callers that need more than geometry -- lane
    headings, speed limits, ids -- do not have to rebuild the chain themselves.
    """
    blocks = [
        b for b in (_get_roadblock(map_api, rid) for rid in route_roadblock_ids) if b is not None
    ]
    blocks = [b for b in blocks if b.interior_edges]
    if not blocks:
        return []

    def lane_points(lane):
        return [(float(p.x), float(p.y)) for p in lane.baseline_path.discrete_path]

    def nearest_lane(candidates, target):
        return min(candidates, key=lambda ln: min(math.dist(p, target) for p in lane_points(ln)))

    chain = [nearest_lane(blocks[0].interior_edges, (float(start_xy[0]), float(start_xy[1])))]
    for block in blocks[1:]:
        outgoing = {edge.id for edge in (getattr(chain[-1], "outgoing_edges", None) or [])}
        successors = [lane for lane in block.interior_edges if lane.id in outgoing]
        # No link (a map gap, or the previous roadblock feeds a different lane): fall back to
        # whichever lane starts nearest the chain's current end, keeping the path continuous.
        chain.append(
            successors[0]
            if successors
            else nearest_lane(block.interior_edges, lane_points(chain[-1])[-1])
        )
    return chain


def build_route_centerline(map_api, route_roadblock_ids, start_xy) -> List[Tuple[float, float]]:
    """Lane centerline through the route, for callers with no driven trajectory to compare against.

    The geometry of :func:`route_lane_chain`, deduplicated into a single polyline.

    Only usable with the ``"centerline"`` path check - see ``PATH_DEVIATION_MAX_P95_CENTERLINE_M``.
    """
    centerline: List[Tuple[float, float]] = []
    for lane in route_lane_chain(map_api, route_roadblock_ids, start_xy):
        for point in lane.baseline_path.discrete_path:
            xy = (float(point.x), float(point.y))
            if not centerline or math.dist(centerline[-1], xy) > 1e-6:
                centerline.append(xy)
    return centerline


def _polyline_end_heading(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Direction of arrival at the end of ``points``, or ``None`` if it never moves.

    Measured over a chord of at least ``MIN_HEADING_SPAN_M`` rather than the final segment: a
    stopped ego logs sub-millimetre jitter whose direction is noise, and reading any segment merely
    longer than zero gave headings up to 180 deg out.
    """
    if len(points) < 2:
        return None

    last = points[-1]
    for candidate in reversed(points[:-1]):
        dx, dy = last[0] - candidate[0], last[1] - candidate[1]
        if math.hypot(dx, dy) >= MIN_HEADING_SPAN_M:
            return math.atan2(dy, dx)
    return None


def _goal_heading_on_route(
    map_api,
    route_roadblock_ids,
    goal: Tuple[float, float],
    reference: Sequence[Tuple[float, float]],
) -> Optional[float]:
    """Lane tangent at a goal stored as bare coordinates, for OSRM's destination bearing.

    Must be the lane tangent, not the ego's direction of travel, because that is what
    :func:`build_routing_task_from_scenario` sends - anything else is a different request for the
    same goal. ``project_goal_to_route_centerline`` also needs a heading as an *input* (it rejects
    lanes more than 90 deg away, keeping the goal off the oncoming carriageway), so the driven path
    seeds the lane choice and the lane supplies the answer. The projected position is discarded.
    """
    seed_heading = _polyline_end_heading(reference)
    if seed_heading is None:
        # No direction to disambiguate lanes with; an unconstrained arrival beats a guessed one.
        return None

    _, goal_heading, _ = project_goal_to_route_centerline(
        map_api,
        route_roadblock_ids,
        Point2D(goal[0], goal[1]),
        seed_heading,
    )
    return goal_heading


def build_routing_task_from_route(
    map_api,
    route_roadblock_ids,
    ego_point,
    ego_heading: float,
    goal_xy,
    *,
    map_name: str,
    epsg: int,
    reference_polyline: Optional[Sequence[Tuple[float, float]]],
    reference_kind: str = "trajectory",
    vias=None,
) -> Dict:
    """Routing task for a caller that already has a route, e.g. a live planner.

    Counterpart to :func:`build_routing_task_from_scenario`, producing the same dict
    :func:`route_description_generation.routing.select_route` consumes, so both share one request
    ladder and one set of validation rules. Being handed the route rather than discovering it means:
    the goal is taken as given (``goal == goal_proj``, so ``route_attempts`` drops the rung that
    varies it), the reference path comes from the caller and is trimmed here to the ego-to-goal
    stretch, and the arrival heading is recovered (see ``_goal_heading_on_route``).

    ``reference_kind`` says what the reference was derived from and so which shape check gates the
    result (see ``routing.PATH_CHECKS``): ``"trajectory"`` for a driven path, ``"centerline"`` for one
    from :func:`build_route_centerline`, whose lane-width offset needs a different statistic.

    ``reference_polyline=None`` means no reference at all: ``nuplan_route_length_m`` stays ``None``
    and ``select_route`` issues one unvalidated request instead of ranking rungs on verdicts it
    cannot compute.
    """
    start_proj, start_heading, _ = project_start_to_route_midpoint(
        map_api,
        route_roadblock_ids,
        ego_point,
        float(ego_heading),
    )
    start = (float(ego_point.x), float(ego_point.y))
    goal = (float(goal_xy[0]), float(goal_xy[1]))

    reference = trim_polyline(reference_polyline or [], start_xy=start, goal_xy=goal)

    return {
        "map_name": map_name,
        "start": start,
        "start_proj": (float(start_proj[0]), float(start_proj[1])),
        # Equal by construction: the goal is taken as given, so there is no second goal to try.
        "goal": goal,
        "goal_proj": goal,
        "epsg": int(epsg),
        "start_heading": start_heading,
        "goal_heading": _goal_heading_on_route(map_api, route_roadblock_ids, goal, reference),
        "route_vias": vias,
        "reference_kind": reference_kind,
        "route_roadblock_ids": list(route_roadblock_ids or []),
        "route_polyline": reference,
        # None, not 0.0: an unknown length must not read as a route that agrees with nothing.
        "nuplan_route_length_m": polyline_length_m(reference) if reference else None,
    }


def build_routing_task_from_scenario(
    scenario,
    route_goal_horizon_s: int,
    sample_frequency_hz: int,
    interplan_goals_by_token: Optional[Dict] = None,
    include_route_vias: bool = True,
) -> Dict:
    map_name = scenario._map_name
    ego_state = scenario.initial_ego_state

    long_term_future_trajectory = list(
        scenario.get_ego_future_trajectory(
            iteration=0,
            num_samples=sample_frequency_hz * route_goal_horizon_s,
            time_horizon=route_goal_horizon_s,
        )
    )
    if not long_term_future_trajectory:
        raise RuntimeError(
            f"Scenario {scenario.token} has no future trajectory samples for routing."
        )
    future_trajectory_length = len(long_term_future_trajectory) / sample_frequency_hz
    goal = long_term_future_trajectory[-1]

    # The route is the roadblocks the ego actually drove through over the routing horizon, not a
    # graph search toward the goal: the branch at every fork is then correct by construction, and
    # the goal is on the route by definition (it is the last sampled pose).
    driven_states = [ego_state] + long_term_future_trajectory
    route_roadblock_ids = build_route_roadblock_ids_from_trajectory(scenario.map_api, driven_states)
    route_roadblock_ids = _append_goal_roadblock(
        scenario.map_api,
        route_roadblock_ids,
        goal.rear_axle.point,
        float(goal.rear_axle.heading),
    )

    start_proj, start_heading, _ = project_start_to_route_midpoint(
        scenario.map_api,
        route_roadblock_ids,
        ego_state.rear_axle.point,
        float(ego_state.rear_axle.heading),
    )
    start_proj = (float(start_proj[0]), float(start_proj[1]))
    start = (float(ego_state.rear_axle.point.x), float(ego_state.rear_axle.point.y))

    goal_proj, goal_heading, _ = project_goal_to_route_centerline(
        scenario.map_api,
        route_roadblock_ids,
        goal.rear_axle.point,
        float(goal.rear_axle.heading),
    )
    goal_proj = (float(goal_proj[0]), float(goal_proj[1]))
    goal = (float(goal.rear_axle.point.x), float(goal.rear_axle.point.y))

    routing_task = {
        "token": str(scenario.token),
        "scenario_type": str(scenario.scenario_type),
        "map_name": map_name,
        "start": start,
        "start_proj": start_proj,
        "goal": goal,
        "goal_proj": goal_proj,
        "epsg": MAP_EPSG[map_name],
        "routing_horizon_s": float(future_trajectory_length),
        "start_heading": start_heading,
        "goal_heading": goal_heading,
        "route_roadblock_ids": route_roadblock_ids,
        "route_connectivity_gaps": route_connectivity_gaps(scenario.map_api, route_roadblock_ids),
        "nuplan_route_length_m": trajectory_arc_length_m(driven_states),
        "route_polyline": ego_state_points(driven_states),
    }
    if include_route_vias:
        projected_goal_point = Point2D(goal_proj[0], goal_proj[1])
        routing_task["route_vias"] = build_route_vias(
            scenario.map_api,
            route_roadblock_ids,
            ego_state.rear_axle.point,
            goal_point=projected_goal_point,
        )

    if interplan_goals_by_token is not None:
        routing_task["interplan_goals"] = interplan_goals_by_token.get(
            str(scenario.token),
            {
                "left": None,
                "right": None,
                "straight": None,
            },
        )
    return routing_task


def build_routing_task_from_alternative_route(
    scenario,
    route_roadblock_ids,
    goal_position,
    *,
    token: Optional[str] = None,
    route_goal_horizon_s: int = 60,
    include_route_vias: bool = True,
) -> Dict:
    """Routing task for a scenario driving a *counterfactual* route.

    Sits between the two existing builders. :func:`build_routing_task_from_scenario` discovers the
    route from the ego's driven trajectory, so by construction it cannot describe a route the ego
    did not take. :func:`build_routing_task_from_route` accepts a route, but expects the caller to
    have already projected the goal, built a reference polyline and chosen vias. This does that
    preparation for an alternative route, mirroring what ``flow_drive.planner.planner`` does at
    inference time under ``--alternative_routing`` -- so an offline alternative-route dataset and
    the live planner describe the same route in the same way.

    The reference polyline is the route's lane centerline rather than a driven path (there is no
    driven path for a route nobody drove), which also selects the ``"centerline"`` shape check in
    :data:`route_description_generation.routing.PATH_CHECKS`.

    Args:
        scenario: a loaded nuPlan scenario; only its map, initial ego state and metadata are read.
        route_roadblock_ids: the alternative route.
        goal_position: ``(x, y)`` or ``(x, y, heading)`` in the global map frame.
        token: dataset row token. Defaults to the scenario's own token; pass the alternative's
            token when several alternatives of one scenario go into the same dataset, or they will
            overwrite each other in the index.
        route_goal_horizon_s: recorded as ``routing_horizon_s`` on the row.
    """
    map_api = scenario.map_api
    map_name = scenario._map_name
    ego_state = scenario.initial_ego_state
    ego_point = ego_state.rear_axle.point
    ego_heading = float(ego_state.rear_axle.heading)

    route_roadblock_ids = list(route_roadblock_ids)
    goal_heading = float(goal_position[2]) if len(goal_position) > 2 else ego_heading
    goal_xy, _, _ = project_goal_to_route_centerline(
        map_api,
        route_roadblock_ids,
        Point2D(float(goal_position[0]), float(goal_position[1])),
        goal_heading,
    )

    reference_polyline = build_route_centerline(
        map_api, route_roadblock_ids, (float(ego_point.x), float(ego_point.y))
    )

    vias = (
        build_route_vias(
            map_api,
            route_roadblock_ids,
            ego_point,
            goal_point=Point2D(float(goal_xy[0]), float(goal_xy[1])),
        )
        if include_route_vias
        else None
    )

    task = build_routing_task_from_route(
        map_api,
        route_roadblock_ids,
        ego_point,
        ego_heading,
        goal_xy,
        map_name=map_name,
        epsg=MAP_EPSG[map_name],
        reference_polyline=reference_polyline,
        reference_kind="centerline",
        vias=vias,
    )
    # process_routing_task reads these off the task; build_routing_task_from_route leaves them out
    # because its original caller (the live planner) never writes a dataset row.
    task["token"] = str(token if token is not None else scenario.token)
    task["scenario_type"] = str(scenario.scenario_type)
    task["routing_horizon_s"] = float(route_goal_horizon_s)
    task["route_connectivity_gaps"] = route_connectivity_gaps(map_api, route_roadblock_ids)
    return task
