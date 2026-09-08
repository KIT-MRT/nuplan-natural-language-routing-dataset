"""Building a routing dataset from nuPlan scenarios.

Owns the parallel scenario -> route pipeline and the writing of the JSONL + SQLite pair that
:mod:`route_description_generation.dataset_index` reads back.
"""

import json
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, cast

from tqdm import tqdm

from route_description_generation.interplan import (
    InterplanGoals,
    load_interplan_modification_goals,
    load_interplan_tokens,
    resolve_interplan_yaml_paths,
)
from route_description_generation.routing import (
    build_and_route_alternative,
    build_and_route_scenario,
    init_scenario_worker,
)
from route_description_generation.scenario_loading import build_scenarios

# Commit the index and flush the data file this often, so a long run that dies partway still
# leaves a readable dataset behind.
_FLUSH_EVERY = 1_000


class DatasetPaths(NamedTuple):
    data_file: Path
    index_file: Path


class BuildSummary(NamedTuple):
    processed: int
    written: int
    valid: int
    failed: int
    without_description: int
    paths: DatasetPaths


def has_route_description(row: Dict) -> bool:
    """Whether a result row carries a usable route description.

    OSRM answering anything other than ``Ok`` yields an empty description rather than an error, so
    such a row reaches the dataset looking like any other failed length check unless it is counted.
    """
    description = (row.get("routing_data") or {}).get("route_description") or ""
    return bool(description.strip())


def chunked(items: Sequence, chunk_size: int) -> Iterator[Sequence]:
    """Yield consecutive slices of ``items`` of at most ``chunk_size``."""
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


class _DatasetWriter:
    """Append-only writer for the JSONL + SQLite offset-index pair.

    Every row's byte offset goes into the index as it is written, which is what lets consumers
    fetch a single token in O(1) without parsing the whole file.
    """

    def __init__(self, paths: DatasetPaths):
        self.paths = paths
        self._connection = sqlite3.connect(paths.index_file)
        self._cursor = self._connection.cursor()
        self._cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS routes (
                token TEXT PRIMARY KEY,
                offset INTEGER NOT NULL
            )
            """
        )
        self._connection.commit()
        # Held open for the writer's lifetime; close() is driven by the context manager.
        self._data = open(paths.data_file, "a", buffering=1)  # noqa: SIM115

    def write(self, row: Dict) -> None:
        offset = self._data.tell()
        self._data.write(json.dumps(row) + "\n")
        self._cursor.execute(
            "INSERT OR REPLACE INTO routes (token, offset) VALUES (?, ?)",
            (row["token"], offset),
        )

    def flush(self) -> None:
        self._connection.commit()
        self._data.flush()

    def close(self) -> None:
        self._connection.commit()
        self._data.close()
        self._connection.close()

    def __enter__(self) -> "_DatasetWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def resolve_interplan_context(
    split: Optional[str],
    enable_interplan_variants: bool,
    scenario_tokens: Optional[List[str]],
) -> tuple:
    """Resolve interplan token filters and goal variants for the selected split.

    Returns ``(scenario_tokens, interplan_goals_by_token)``. Both are passed through unchanged
    when interplan is not in play.
    """
    if not (split == "interplan" or enable_interplan_variants):
        return scenario_tokens, None

    benchmark_yaml, modifications_yaml = resolve_interplan_yaml_paths(None, None)

    if split == "interplan":
        if benchmark_yaml is not None and modifications_yaml is not None:
            interplan_tokens = load_interplan_tokens(
                benchmark_yaml=benchmark_yaml, modifications_yaml=modifications_yaml
            )
            if scenario_tokens is None:
                scenario_tokens = interplan_tokens
                print(
                    f"Loaded {len(interplan_tokens):,} interplan tokens from benchmark+modification yaml"
                )
            else:
                scenario_tokens = cast(
                    List[str], sorted(set(scenario_tokens).intersection(interplan_tokens))
                )
                print(
                    "Intersected scenario token filter with interplan tokens: "
                    f"{len(scenario_tokens):,} remain"
                )
        else:
            print(
                "Split 'interplan' selected without interplan YAML files. "
                "Proceeding with split log-name filtering only."
            )

    interplan_goals_by_token: Optional[Dict[str, InterplanGoals]] = None
    if modifications_yaml is None:
        if enable_interplan_variants:
            raise ValueError(
                "--enable-interplan-variants requires interplan to be installed with a "
                "discoverable modifications yaml"
            )
    else:
        interplan_goals_by_token = load_interplan_modification_goals(modifications_yaml)
        print(f"Loaded interplan goal variants for {len(interplan_goals_by_token):,} tokens")

    return scenario_tokens, interplan_goals_by_token


def dataset_paths(output_dir: Path, split: Optional[str], prefix: str) -> DatasetPaths:
    """Locate the dataset file pair for a split/prefix combination."""
    stem = f"{split}_{prefix}" if split else prefix
    return DatasetPaths(
        data_file=output_dir / f"{stem}_data.jsonl",
        index_file=output_dir / f"{stem}_index.sqlite",
    )


def route_scenarios(
    scenarios: Sequence,
    writer: _DatasetWriter,
    *,
    workers: int,
    chunk_size: int,
    worker_settings: tuple,
) -> BuildSummary:
    """Route every scenario in parallel and write the results through ``writer``."""
    processed = 0
    written = 0
    valid = 0
    failed = 0
    without_description = 0
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_scenario_worker,
            initargs=worker_settings,
        ) as executor,
        tqdm(
            total=len(scenarios),
            desc="Processing scenarios",
            unit="scenarios",
            miniters=1000,
        ) as progress,
    ):
        # Chunked so at most chunk_size futures (and their results) are alive at once; within
        # a chunk, building and routing overlap freely across workers.
        for chunk in chunked(scenarios, chunk_size):
            futures = {
                executor.submit(build_and_route_scenario, scenario): getattr(
                    scenario, "token", "unknown"
                )
                for scenario in chunk
            }
            for future in as_completed(futures):
                try:
                    row = future.result()
                    writer.write(row)
                    written += 1
                    if row.get("routing_data", {}).get("valid_route"):
                        valid += 1
                    if not has_route_description(row):
                        without_description += 1
                except Exception as error:
                    # The scenario produces no row at all, so it would otherwise vanish from the
                    # dataset with only this line to say so; it is counted for the summary.
                    failed += 1
                    tqdm.write(f"ERROR processing scenario {futures[future]}: {error}")
                finally:
                    processed += 1
                    progress.update(1)
                    if processed % _FLUSH_EVERY == 0:
                        writer.flush()

    return BuildSummary(
        processed=processed,
        written=written,
        valid=valid,
        failed=failed,
        without_description=without_description,
        paths=writer.paths,
    )


def build_dataset(
    *,
    data_path: str,
    map_path: str,
    output_dir: Path,
    split: Optional[str],
    output_prefix: str = "lg_routing",
    map_version: str = "nuplan-maps-v1.0",
    log_names: Optional[List[str]] = None,
    scenario_tokens: Optional[List[str]] = None,
    total_scenarios: Optional[int] = None,
    scenarios_per_type: Optional[int] = None,
    shuffle_scenarios: bool = False,
    enable_interplan_variants: bool = False,
    workers: int = 12,
    chunk_size: int = 1000,
    routing_service: str = "osrm",
    route_goal_horizon_s: int = 60,
    sample_frequency_hz: int = 10,
    extraction_offset: float = 0.0,
) -> BuildSummary:
    """Build a routing dataset and write it to ``output_dir``."""
    scenario_tokens, interplan_goals_by_token = resolve_interplan_context(
        split, enable_interplan_variants, scenario_tokens
    )

    print("Loading scenarios with nuPlan-devkit...")
    scenarios = build_scenarios(
        data_path=data_path,
        map_path=map_path,
        map_version=map_version,
        log_names=log_names,
        scenario_tokens=scenario_tokens,
        total_scenarios=total_scenarios,
        scenarios_per_type=scenarios_per_type,
        shuffle_scenarios=shuffle_scenarios,
        extraction_offset=extraction_offset,
    )
    print(f"Found {len(scenarios):,} scenarios")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = dataset_paths(output_dir, split, output_prefix)

    with _DatasetWriter(paths) as writer:
        summary = route_scenarios(
            scenarios,
            writer,
            workers=workers,
            chunk_size=chunk_size,
            worker_settings=(
                route_goal_horizon_s,
                sample_frequency_hz,
                interplan_goals_by_token,
                routing_service,
            ),
        )

    print(f"Done. Processed {summary.processed:,} scenarios.")
    if summary.failed:
        print(
            f"FAILED: {summary.failed:,} scenarios raised and produced no dataset row "
            "(see the ERROR lines above)"
        )
    if summary.without_description:
        print(
            f"WARNING: {summary.without_description:,} rows have an empty route description "
            "(OSRM returned no route for every attempt)"
        )
    print(f"Valid routes: {summary.valid:,} / {summary.written:,} written")
    print(f"Data saved to {paths.data_file}")
    print(f"Index saved to {paths.index_file}")
    return summary


def route_alternatives(
    items: Sequence,
    writer: _DatasetWriter,
    *,
    workers: int,
    chunk_size: int,
    worker_settings: tuple,
) -> BuildSummary:
    """Route ``(scenario, alternative)`` pairs in parallel and write the results.

    The scenario-driven twin of this is :func:`route_scenarios`; the only differences are the
    worker function and where the progress label's token comes from, so the flush cadence, the
    chunking rationale and the per-row failure handling are all as documented there.
    """
    processed = 0
    written = 0
    valid = 0
    failed = 0
    without_description = 0
    with (
        ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_scenario_worker,
            initargs=worker_settings,
        ) as executor,
        tqdm(
            total=len(items),
            desc="Routing alternatives",
            unit="routes",
            miniters=100,
        ) as progress,
    ):
        for chunk in chunked(items, chunk_size):
            futures = {
                executor.submit(build_and_route_alternative, item): item[1].get("alt_token", "unknown")
                for item in chunk
            }
            for future in as_completed(futures):
                try:
                    row = future.result()
                    writer.write(row)
                    written += 1
                    if row.get("routing_data", {}).get("valid_route"):
                        valid += 1
                    if not has_route_description(row):
                        without_description += 1
                except Exception as error:
                    failed += 1
                    tqdm.write(f"ERROR routing alternative {futures[future]}: {error}")
                finally:
                    processed += 1
                    progress.update(1)
                    if processed % _FLUSH_EVERY == 0:
                        writer.flush()

    return BuildSummary(
        processed=processed,
        written=written,
        valid=valid,
        failed=failed,
        without_description=without_description,
        paths=writer.paths,
    )


def build_dataset_from_routes(
    items: Sequence,
    output_dir: Path,
    *,
    split: Optional[str] = None,
    output_prefix: str = "lg_routing",
    workers: int = 12,
    chunk_size: int = 1000,
    routing_service: str = "osrm",
    route_goal_horizon_s: int = 60,
    sample_frequency_hz: int = 10,
) -> BuildSummary:
    """Build a routing dataset for routes supplied by the caller.

    The counterpart to :func:`build_dataset`, which discovers each scenario's route from the driven
    trajectory. Here the route comes in with the scenario, which is what makes it usable for
    counterfactual routes -- alternative routes, re-routings, anything the ego did not actually
    drive.

    Args:
        items: ``(scenario, alternative)`` pairs. ``alternative`` is a dict with at least
            ``route_roadblock_ids``, ``goal_position`` and ``alt_token``; ``source_token``,
            ``alt_index`` and ``instruction`` are carried onto the row when present.
        output_dir: directory for the ``<prefix>_data.jsonl`` / ``<prefix>_index.sqlite`` pair.

    Note that the writer appends, so calling this twice against one directory extends the dataset
    rather than replacing it -- and the SQLite index upserts by token, so a re-run of the same
    tokens leaves the index pointing at the newest row.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = dataset_paths(output_dir, split, output_prefix)

    with _DatasetWriter(paths) as writer:
        summary = route_alternatives(
            items,
            writer,
            workers=workers,
            chunk_size=chunk_size,
            worker_settings=(
                route_goal_horizon_s,
                sample_frequency_hz,
                None,  # interplan goals: not applicable to caller-supplied routes
                routing_service,
            ),
        )

    print(f"Done. Routed {summary.processed:,} alternatives.")
    if summary.failed:
        print(f"FAILED: {summary.failed:,} produced no dataset row (see the ERROR lines above)")
    if summary.without_description:
        print(
            f"WARNING: {summary.without_description:,} rows have an empty route description "
            "(OSRM returned no route for every attempt)"
        )
    print(f"Valid routes: {summary.valid:,} / {summary.written:,} written")
    print(f"Data saved to {paths.data_file}")
    print(f"Index saved to {paths.index_file}")
    return summary
