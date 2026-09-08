"""Polyline decoding and path-agreement measurement.

Route length agreement is a weak proxy for "OSRM describes the path the ego drove": a route can
match in length while following an entirely different street. These helpers compare the two
*geometries* instead, which detects that class directly.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]

# Spacing (metres) both polylines are resampled to before they are compared. OSRM's `overview=full`
# geometry only carries vertices where the road bends - the median val14 route has about seven of
# them - so comparing raw vertices would measure how bendy a road is rather than how wrong a route
# is. Resampling makes the deviation independent of vertex density.
PATH_COMPARISON_SPACING_M = 5.0


def decode_polyline(encoded: str, precision: int = 5) -> List[Point]:
    """Decode an encoded polyline into ``(lon, lat)`` pairs.

    OSRM returns geometry in this format whenever ``geometries`` is left at its default, with
    ``precision=5``. Coordinates are emitted lon-first so they can be handed straight to a pyproj
    ``Transformer`` built with ``always_xy=True``.
    """
    coordinates: List[Point] = []
    scale = float(10**precision)
    index = 0
    lat = 0
    lon = 0
    length = len(encoded)

    while index < length:
        for is_latitude in (True, False):
            shift = 0
            result = 0
            while index < length:
                byte = ord(encoded[index]) - 63
                index += 1
                result |= (byte & 0x1F) << shift
                shift += 5
                if byte < 0x20:
                    break
            delta = ~(result >> 1) if result & 1 else result >> 1
            if is_latitude:
                lat += delta
            else:
                lon += delta
        coordinates.append((lon / scale, lat / scale))

    return coordinates


def resample_polyline(
    points: Sequence[Point], spacing_m: float = PATH_COMPARISON_SPACING_M
) -> List[Point]:
    """Subdivide ``points`` so no gap exceeds ``spacing_m``, keeping every original vertex.

    Vertices are preserved rather than resampled at even intervals: walking at a fixed stride skips
    whichever vertices fall mid-stride, which cuts corners and - at a turnaround - can drop the apex
    entirely, inventing a deviation that the path does not have. Subdividing each segment keeps the
    geometry exact while still bounding the gap between samples.

    Expects a projected (metre) frame. Returns the input unchanged when it is too short to walk.
    """
    path = [(float(x), float(y)) for x, y in points]
    if len(path) < 2 or spacing_m <= 0:
        return path

    resampled: List[Point] = [path[0]]
    for start, end in zip(path, path[1:]):
        segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
        if segment_length <= 0.0:
            continue
        steps = max(int(math.ceil(segment_length / spacing_m)), 1)
        for step in range(1, steps + 1):
            ratio = step / steps
            resampled.append(
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
            )
    return resampled


def polyline_length_m(points: Sequence[Point]) -> float:
    """Arc length along a polyline in a projected (metre) frame."""
    path = [(float(x), float(y)) for x, y in points]
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])))


def _nearest_vertex_index(path: Sequence[Point], target: Point) -> int:
    return min(
        range(len(path)), key=lambda i: math.hypot(path[i][0] - target[0], path[i][1] - target[1])
    )


def trim_polyline(
    points: Sequence[Point],
    start_xy: Optional[Point] = None,
    goal_xy: Optional[Point] = None,
) -> List[Point]:
    """Cut ``points`` down to the stretch between ``start_xy`` and ``goal_xy``.

    Endpoints snap to their nearest *vertex*, so the result stays a sub-sequence of the input and
    every original vertex survives - the property ``resample_polyline`` depends on. ``None`` leaves
    that end alone. A goal snapping at or before the start (ego reached it, or the path doubles back)
    collapses to the single start vertex rather than reversing.
    """
    path = [(float(x), float(y)) for x, y in points]
    if not path:
        return []

    first = _nearest_vertex_index(path, start_xy) if start_xy is not None else 0
    last = _nearest_vertex_index(path, goal_xy) if goal_xy is not None else len(path) - 1
    return path[first : max(last, first) + 1]


def _point_to_polyline_distances(points: np.ndarray, path: np.ndarray) -> np.ndarray:
    """Shortest distance from every point to the *segments* of ``path``.

    Segment distance rather than vertex distance, because a sparse polyline's vertices can sit far
    apart along a road that the other path follows exactly.
    """
    if len(path) == 1:
        return np.linalg.norm(points - path[0], axis=1)

    starts = path[:-1]
    ends = path[1:]
    segments = ends - starts
    lengths_squared = np.einsum("ij,ij->i", segments, segments)
    # Degenerate (zero-length) segments would divide by zero; clamping t to 0 makes them behave as
    # the start vertex, which is the correct limit.
    safe_lengths = np.where(lengths_squared > 0.0, lengths_squared, 1.0)

    offsets = points[:, None, :] - starts[None, :, :]
    t = np.einsum("ijk,jk->ij", offsets, segments) / safe_lengths[None, :]
    t = np.clip(np.where(lengths_squared[None, :] > 0.0, t, 0.0), 0.0, 1.0)

    closest = starts[None, :, :] + t[:, :, None] * segments[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1)


def path_deviation(
    path_a: Sequence[Point],
    path_b: Sequence[Point],
    spacing_m: float = PATH_COMPARISON_SPACING_M,
) -> Optional[Dict[str, float]]:
    """Symmetric geometric deviation between two polylines in a projected (metre) frame.

    Both directions are measured and the *worse* of the two is reported: measuring only a-to-b
    misses a stretch that ``path_b`` skipped, and only b-to-a misses a detour ``path_b`` added.

    Every statistic takes the worse direction rather than pooling the two sets of distances,
    because pooling weights each direction by its point count. A driven trajectory sampled at 10 Hz
    resamples to ~600 points where a 600 m OSRM route resamples to ~120, so a pooled mean is
    dominated 5:1 by the driven-to-route direction and reads far lower than either direction: a
    route covering the driven path *plus* a 190 m excursion scored a pooled mean of 4.5 m against a
    route-to-driven mean of 14.1 m.

    Returns ``None`` when either path is empty.
    """
    a = resample_polyline(path_a, spacing_m)
    b = resample_polyline(path_b, spacing_m)
    if not a or not b:
        return None

    forward = _point_to_polyline_distances(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    backward = _point_to_polyline_distances(np.asarray(b, dtype=float), np.asarray(a, dtype=float))

    return {
        "max_m": float(max(forward.max(), backward.max())),
        "p95_m": float(max(np.percentile(forward, 95.0), np.percentile(backward, 95.0))),
        "mean_m": float(max(forward.mean(), backward.mean())),
    }
