"""Random access into a generated routing dataset.

A dataset is a pair of files: ``<prefix>_data.jsonl`` holding one JSON row per scenario, and
``<prefix>_index.sqlite`` mapping each token to that row's byte offset. The offset index is what
lets consumers read a single scenario in O(1) without parsing the whole JSONL.
"""

import atexit
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from tqdm import tqdm

# Open file and connection handles, cached per process. PyTorch ``DataLoader`` workers with
# ``persistent_workers=True`` call load_by_token once per sample, so reopening each time would
# dominate the read cost.
_HANDLE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _close_cached_handles() -> None:
    for handles in list(_HANDLE_CACHE.values()):
        for key in ("file", "conn"):
            handle = handles.get(key)
            if handle is None:
                continue
            try:
                handle.close()
            except Exception:
                # Interpreter shutdown can close these first; nothing useful to do about it.
                pass
    _HANDLE_CACHE.clear()


atexit.register(_close_cached_handles)


def _cached_handles(sqlite_index_file: str, json_data_file: str) -> Dict[str, Any]:
    key = (sqlite_index_file, json_data_file)
    if key not in _HANDLE_CACHE:
        connection = sqlite3.connect(sqlite_index_file)
        _HANDLE_CACHE[key] = {
            "conn": connection,
            "cur": connection.cursor(),
            # Cached for the process lifetime and closed by the atexit hook above.
            "file": open(json_data_file, "r"),  # noqa: SIM115
        }
    return _HANDLE_CACHE[key]


def load_by_token(
    sqlite_index_file: str, json_data_file: str, token: str
) -> Optional[Dict[str, Any]]:
    """Return the dataset row for ``token``, or ``None`` when it is not in the index."""
    handles = _cached_handles(sqlite_index_file, json_data_file)
    handles["cur"].execute("SELECT offset FROM routes WHERE token = ?", (token,))
    row = handles["cur"].fetchone()
    if row is None:
        return None

    handles["file"].seek(row[0])
    return json.loads(handles["file"].readline())


def find_token_by_route_start(json_data_file: str, route_start: list) -> Optional[str]:
    """Return the token whose ``route_start`` is closest to ``route_start``.

    A linear scan, unlike :func:`load_by_token` - there is no index on position. Returns the exact
    match when one exists, otherwise the nearest by Euclidean distance, or ``None`` for an empty
    dataset.
    """
    closest_token = None
    min_distance = float("inf")

    with open(json_data_file, "r") as data:
        for line in data:
            row = json.loads(line)
            candidate = row["routing_data"]["route_start"]
            if candidate == route_start:
                return row["token"]

            distance = sum((a - b) ** 2 for a, b in zip(route_start, candidate)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_token = row["token"]

    return closest_token


def _iter_lines(json_data_file: Union[str, Path]) -> Iterator[Tuple[int, str]]:
    """Yield ``(line_number, line)`` for every non-blank line."""
    with open(json_data_file, "r", encoding="utf-8") as data:
        for line_number, line in enumerate(data, start=1):
            line = line.strip()
            if line:
                yield line_number, line


def _parse_row(
    line: str, line_number: int, json_data_file: Union[str, Path]
) -> Optional[Dict[str, Any]]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON at line {line_number} in {json_data_file}: {error}"
        ) from error
    return row if isinstance(row, dict) else None


def iter_rows(json_data_file: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """Yield every row of a dataset JSONL, skipping blanks.

    A full scan, for the cases where the offset index does not help - selecting rows by a property
    rather than by token.
    """
    for line_number, line in _iter_lines(json_data_file):
        row = _parse_row(line, line_number, json_data_file)
        if row is not None:
            yield row


def collect_tokens(
    json_data_file: Union[str, Path],
    *,
    valid_route: bool,
    show_progress: bool = False,
) -> List[str]:
    """Tokens whose ``valid_route`` is exactly ``valid_route``, in file order, without duplicates.

    Matched strictly against ``True``/``False`` rather than by truthiness, so a row that predates
    the field - or failed before it was written - is never mistaken for either answer.

    ``show_progress`` reports rows scanned as they go. There is no total to count towards without
    reading the file twice, so it is a running count rather than a percentage - worth having on a
    multi-gigabyte train dataset, noise on a small one.
    """
    needle = _valid_route_needle(json_data_file, valid_route)

    tokens: List[str] = []
    seen: Set[str] = set()
    lines = tqdm(
        _iter_lines(json_data_file),
        desc="Scanning dataset",
        unit="row",
        disable=not show_progress,
    )
    for line_number, line in lines:
        # Parsing every row means decoding the whole embedded OSRM response for each one, which
        # dominates the scan. When the file's spacing is known, a substring test rejects the vast
        # majority of lines first; anything it lets through is still parsed and checked properly,
        # so the fast path can never disagree with the slow one.
        if needle is not None and needle not in line:
            continue
        row = _parse_row(line, line_number, json_data_file)
        if row is None:
            continue
        token = row.get("token")
        if not isinstance(token, str) or not token or token in seen:
            continue
        if (row.get("routing_data") or {}).get("valid_route") is valid_route:
            tokens.append(token)
            seen.add(token)
    return tokens


def _valid_route_needle(json_data_file: Union[str, Path], valid_route: bool) -> Optional[str]:
    """A substring every matching row must contain, or ``None`` when it cannot be established.

    The spacing is taken from the file itself rather than assumed, because a guess that is wrong
    would silently reject every row instead of merely being slow.
    """
    literal = "true" if valid_route else "false"
    for _, line in _iter_lines(json_data_file):
        for separator in ('"valid_route": ', '"valid_route":'):
            if separator in line:
                return f"{separator}{literal}"
        return None
    return None
