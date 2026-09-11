"""Random access into a generated routing dataset.

A dataset is a pair of files: ``<prefix>_data.jsonl`` holding one JSON row per scenario, and
``<prefix>_index.sqlite`` mapping each token to that row's byte offset. The offset index is what
lets consumers read a single scenario in O(1) without parsing the whole JSONL.
"""

import atexit
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union
from urllib.parse import quote

from tqdm import tqdm

# Open file and connection handles, cached per process. PyTorch ``DataLoader`` workers with
# ``persistent_workers=True`` call load_by_token once per sample, so reopening each time would
# dominate the read cost.
_HANDLE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}

#: How many times a read is retried after a transient filesystem error before giving up. These
#: datasets are routinely read from network storage by a hundred-plus DataLoader workers at once,
#: where a single failed RPC surfaces as ``sqlite3.OperationalError: disk I/O error`` and kills the
#: worker -- and with it a training run that may be dozens of epochs in. A blip is worth a retry.
_IO_RETRY_ATTEMPTS = 3
#: Seconds to wait before the first retry; doubled on each subsequent one.
_IO_RETRY_BACKOFF_S = 0.25


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


def _immutable_uri(sqlite_index_file: str) -> str:
    """SQLite URI opening the index read-only with locking disabled.

    ``immutable=1`` promises SQLite the file will not change while it is open, which lets it skip
    every lock and every change-counter check. On a POSIX filesystem that only saves a little; on
    NFS mounted with ``local_lock=none`` each of those checks is a network round-trip, and a
    ``SELECT`` by token costs ~2 ms instead of ~7 us. It also removes the lock traffic that makes
    the read fail under load in the first place.

    The promise is real: regenerating an index while a reader process holds it open gives that
    reader stale or corrupt pages. Builders (``dataset_builder``, ``cli.modify_dataset``) open
    their own read-write connections and are unaffected; only start a rebuild once readers are done.
    """
    return f"file:{quote(str(Path(sqlite_index_file).resolve()))}?immutable=1"


def _cached_handles(sqlite_index_file: str, json_data_file: str) -> Dict[str, Any]:
    key = (sqlite_index_file, json_data_file)
    if key not in _HANDLE_CACHE:
        # An immutable connection to a missing file opens happily and only fails later with
        # "no such table: routes", which sends the reader looking for the wrong problem.
        if not Path(sqlite_index_file).exists():
            raise FileNotFoundError(f"routing index not found: {sqlite_index_file}")
        connection = sqlite3.connect(_immutable_uri(sqlite_index_file), uri=True)
        _HANDLE_CACHE[key] = {
            "conn": connection,
            "cur": connection.cursor(),
            # Cached for the process lifetime and closed by the atexit hook above.
            "file": open(json_data_file, "r"),  # noqa: SIM115
        }
    return _HANDLE_CACHE[key]


def _drop_cached_handles(key: Tuple[str, str]) -> None:
    """Forget one cache entry so the next call reopens it, closing what can still be closed."""
    handles = _HANDLE_CACHE.pop(key, None)
    if handles is None:
        return
    for name in ("file", "conn"):
        handle = handles.get(name)
        if handle is None:
            continue
        try:
            handle.close()
        except Exception:
            # The handle is being discarded either way; a failure to close it changes nothing.
            pass


def _read_by_token(
    sqlite_index_file: str, json_data_file: str, token: str
) -> Optional[Dict[str, Any]]:
    """One attempt at :func:`load_by_token`, without the retry."""
    handles = _cached_handles(sqlite_index_file, json_data_file)
    handles["cur"].execute("SELECT offset FROM routes WHERE token = ?", (token,))
    row = handles["cur"].fetchone()
    if row is None:
        return None

    handles["file"].seek(row[0])
    return json.loads(handles["file"].readline())


def load_by_token(
    sqlite_index_file: str, json_data_file: str, token: str
) -> Optional[Dict[str, Any]]:
    """Return the dataset row for ``token``, or ``None`` when it is not in the index.

    Retries a transient filesystem failure :data:`_IO_RETRY_ATTEMPTS` times, dropping the cached
    handles first so the retry reconnects rather than reusing a connection the error may have left
    unusable. Only I/O errors are retried -- a missing index, a malformed row or an absent token
    are answers, not blips, and are raised or returned immediately.
    """
    key = (sqlite_index_file, json_data_file)
    backoff = _IO_RETRY_BACKOFF_S
    for attempt in range(_IO_RETRY_ATTEMPTS):
        try:
            return _read_by_token(sqlite_index_file, json_data_file, token)
        except (sqlite3.OperationalError, OSError) as error:
            # FileNotFoundError is an OSError, but it is a configuration mistake rather than a
            # blip: retrying it just delays the same failure.
            if isinstance(error, FileNotFoundError) or attempt == _IO_RETRY_ATTEMPTS - 1:
                raise
            _drop_cached_handles(key)
            time.sleep(backoff)
            backoff *= 2
    raise AssertionError("unreachable")  # pragma: no cover


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
