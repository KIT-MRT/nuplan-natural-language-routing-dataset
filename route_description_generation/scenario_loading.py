"""Selecting and loading nuPlan scenarios.

Turns CLI filters (split name, log-name list, token file) into the concrete set of scenarios the
rest of the pipeline operates on.
"""

import json
import os
import random
import sqlite3
from pathlib import Path
from typing import List, Optional, Set, cast

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioExtractionInfo,
    ScenarioMapping,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)
from tqdm import tqdm

MAP_EPSG = {
    "us-ma-boston": 32619,
    "us-pa-pittsburgh-hazelwood": 32617,
    "sg-one-north": 32648,
    "us-nv-las-vegas-strip": 32611,
}

SPLIT_TO_LOG_NAMES_FILE = {
    "train": "nuplan_train.json",
    "val": "nuplan_val.json",
    "val14": "nuplan_val.json",
    "interplan": "nuplan_test.json",
}

# Splits served by the trainval databases. Everything else - interPlan today - lives in test.
_TRAINVAL_SPLITS = frozenset({"train", "val", "val14"})

# Layout of a standard nuPlan v1.1 installation below NUPLAN_DATA_ROOT.
_SPLITS_SUBPATH = Path("nuplan-v1.1") / "splits"

# Seconds of run-up nuPlan's simulation builders give the ego before the tagged token. Matches the
# `nuplan_challenge` / `nuplan_eval` scenario mappings, which extract *every* scenario type at
# [15.0, -3.0], and therefore the pose a simulated planner actually starts from.
SIMULATION_EXTRACTION_OFFSET_S = -3.0

# Splits whose datasets are consumed by simulation rather than by training.
_SIMULATION_SPLITS = frozenset({"val", "val14", "interplan"})


class FixedWindowScenarioMapping(ScenarioMapping):
    """A ``ScenarioMapping`` applying one extraction window to *every* scenario type.

    nuPlan keys extraction info by scenario type, but this package loads by token across all types
    and only reads iteration 0, so one window is both sufficient and the only way to get a uniform
    offset. ``initial_ego_state`` moves to ``anchor + extraction_offset`` while
    ``scenario.token``/``get_mission_goal``/``get_route_roadblock_ids`` stay on the anchor.

    ``subsample_ratio`` stays 1.0 despite the simulation's 0.5: the token query selects
    ``((row_num - 1) % interval) == 0`` over a *window-local* row number, so index 0 is the window's
    first token either way.
    """

    def __init__(self, extraction_offset: float) -> None:
        super().__init__({}, None)
        self._extraction_info = ScenarioExtractionInfo(
            scenario_name="rdg_offset",
            # Only iteration 0 is read; the assert just requires this to be positive.
            scenario_duration=abs(extraction_offset) + 1.0,
            extraction_offset=extraction_offset,
            subsample_ratio=1.0,
        )

    def get_extraction_info(self, scenario_type: str) -> ScenarioExtractionInfo:
        return self._extraction_info


def resolve_extraction_offset(split: Optional[str]) -> float:
    """Seconds of run-up to load before the tagged token, derived from ``split``.

    Simulation splits take the simulation's own ``-3.0`` so the dataset's start pose, route and goal
    are anchored where a simulated planner begins. Training splits stay at ``0.0``, the pose their
    cached features are built from.
    """
    return SIMULATION_EXTRACTION_OFFSET_S if split in _SIMULATION_SPLITS else 0.0


def resolve_data_path(split: Optional[str]) -> Optional[Path]:
    """Database directory for ``split``, derived from ``NUPLAN_DATA_ROOT``.

    train/val/val14 are served by ``trainval``; any other split by ``test``. Returns ``None`` when
    there is no split to key off or ``NUPLAN_DATA_ROOT`` is unset, leaving it to the caller to
    demand an explicit path.
    """
    data_root = os.environ.get("NUPLAN_DATA_ROOT")
    if split is None or not data_root:
        return None

    subdirectory = "trainval" if split in _TRAINVAL_SPLITS else "test"
    return Path(data_root) / _SPLITS_SUBPATH / subdirectory


def resolve_maps_path() -> Optional[Path]:
    """nuPlan maps directory from ``NUPLAN_MAPS_ROOT``, or ``None`` when unset."""
    maps_root = os.environ.get("NUPLAN_MAPS_ROOT")
    return Path(maps_root) if maps_root else None


def resolve_scenario_tokens_file(split: Optional[str], res_dir: Path) -> Optional[Path]:
    """The shipped token list for ``split``, when there is one.

    Some splits are *defined* by their token list rather than by log names - val14 and val share
    ``nuplan_val.json``, and only ``res/val14_tokens.txt`` distinguishes the 1118-scenario
    benchmark from the whole val split. Returns ``None`` when no such file ships for the split.
    """
    if split is None:
        return None

    tokens_file = res_dir / f"{split}_tokens.txt"
    return tokens_file if tokens_file.is_file() else None


def get_filter_parameters(
    num_scenarios_per_type: Optional[int],
    limit_total_scenarios: Optional[int],
    shuffle: bool,
    scenario_tokens: Optional[List[str]],
    log_names: Optional[List[str]],
    expand_scenarios: bool = True,
):
    scenario_types = None
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None
    remove_invalid_goals = False
    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


def max_db_filter_tokens() -> int:
    """How many tokens may be pushed into the devkit's SQL token filter.

    SQLite binds one host parameter per token, capped at 999 before 3.32 and 32766 from 3.32 on.
    The margin leaves room for the other bound values in the same query (scenario types, map
    names).
    """
    limit = 32766 if sqlite3.sqlite_version_info >= (3, 32) else 999
    return limit - 256


def load_json_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_token_filter(path: Path) -> List[str]:
    tokens: Set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            token = line.strip()
            if token:
                tokens.add(token)
    return cast(List[str], sorted(tokens))


def resolve_log_names(
    split: Optional[str],
    log_names_json: Optional[Path],
    split_logs_dir: Path,
) -> Optional[List[str]]:
    if log_names_json is not None:
        return load_json_list(log_names_json)

    if split is None:
        return None

    split_file = SPLIT_TO_LOG_NAMES_FILE[split]
    split_path = split_logs_dir / split_file
    if not split_path.exists():
        print(
            f"Split '{split}' selected, but log file not found at {split_path}. "
            "Continuing without log-name filtering."
        )
        return None

    print(f"Using split log names from {split_path}")
    return load_json_list(split_path)


def build_scenarios(
    data_path: str,
    map_path: str,
    map_version: str,
    log_names: Optional[List[str]],
    scenario_tokens: Optional[List[str]],
    total_scenarios: Optional[int],
    scenarios_per_type: Optional[int],
    shuffle_scenarios: bool,
    extraction_offset: float = 0.0,
):
    sensor_root = ""
    db_files = None

    # nuPlan-devkit turns a token filter into `lp.token IN (?,?,...)` with one bind variable per
    # token. Past SQLite's host-parameter limit that query fails outright with "too many SQL
    # variables", so an oversized list is issued as several queries and the results concatenated.
    # Batching keeps the devkit's own filter in charge, so the loaded set matches the requested
    # tokens exactly instead of being narrowed after the fact.
    token_batches = _batch_tokens(scenario_tokens, max_db_filter_tokens())
    batching = len(token_batches) > 1
    if batching:
        print(
            f"Token filter has {len(scenario_tokens or []):,} entries; querying in "
            f"{len(token_batches)} batches to stay within SQLite's variable limit"
        )

    # None at offset 0 so the devkit installs its own default, keeping an unoffset run identical to
    # one built before this option existed.
    scenario_mapping = (
        FixedWindowScenarioMapping(extraction_offset) if extraction_offset != 0.0 else None
    )
    # expand_scenarios decides whether the mapping is consulted at all: the devkit passes
    # scenario_extraction_info=None whenever it is set (nuplan_scenario_filter_utils.py:191),
    # collapsing each scenario to the single anchor frame and silently discarding any offset.
    expand_scenarios = scenario_mapping is None

    builder = NuPlanScenarioBuilder(
        data_path,
        map_path,
        sensor_root,
        db_files,
        map_version,
        scenario_mapping=scenario_mapping,
    )
    worker = SingleMachineParallelExecutor(use_process_pool=True)

    scenarios = []
    loaded_tokens: Set[str] = set()
    # Each batch is a full pass over every log file, so on a large token filter this is the slow
    # part of a run. There is no progress to report inside one pass - the devkit fans out over log
    # files internally - so the bar counts batches, and is pointless when there is only one.
    batch_progress = tqdm(
        token_batches,
        desc="Loading scenarios",
        unit="batch",
        disable=not batching,
    )
    for batch in batch_progress:
        scenario_filter = ScenarioFilter(  # type: ignore[arg-type]
            *get_filter_parameters(
                num_scenarios_per_type=scenarios_per_type,
                # Applied per batch these would each get the full allowance, so the cap and the
                # shuffle are applied once over the combined result instead.
                limit_total_scenarios=None if batching else total_scenarios,
                shuffle=False if batching else shuffle_scenarios,
                scenario_tokens=batch,
                log_names=log_names,
                expand_scenarios=expand_scenarios,
            )
        )
        for scenario in builder.get_scenarios(scenario_filter, worker):
            if scenario.token not in loaded_tokens:
                loaded_tokens.add(scenario.token)
                scenarios.append(scenario)
        batch_progress.set_postfix(scenarios=f"{len(scenarios):,}", refresh=False)

    if scenario_tokens is not None:
        print(f"Loaded {len(loaded_tokens):,} of {len(scenario_tokens):,} requested tokens")
        missing = set(scenario_tokens) - loaded_tokens
        if missing:
            sample = ", ".join(sorted(missing)[:5])
            print(
                f"WARNING: {len(missing):,} requested tokens are not present in this split "
                f"(e.g. {sample})"
            )

    if batching:
        if shuffle_scenarios:
            random.shuffle(scenarios)
        if total_scenarios is not None:
            scenarios = scenarios[:total_scenarios]

    del worker, builder
    return scenarios


def _batch_tokens(
    scenario_tokens: Optional[List[str]], batch_size: int
) -> List[Optional[List[str]]]:
    """Split a token filter into query-sized batches.

    ``None`` (no filter) becomes a single unfiltered query, so callers can treat both cases the
    same way.
    """
    if scenario_tokens is None:
        return [None]
    if len(scenario_tokens) <= batch_size:
        return [scenario_tokens]
    return [
        scenario_tokens[start : start + batch_size]
        for start in range(0, len(scenario_tokens), batch_size)
    ]
