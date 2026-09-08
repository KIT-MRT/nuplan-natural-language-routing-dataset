"""``rdg build-dataset-from-routes`` -- describe routes the ego did not drive.

``rdg build-dataset`` derives each scenario's route from its driven trajectory, so it can only ever
describe what happened. This command takes the routes as input instead, one JSONL row per route,
which is what makes it usable for counterfactual routes -- the alternative routes produced by
``nucontrol`` / ``nuscenario_generation``, or any other externally chosen route.

Input rows need ``source_token``, ``route_roadblock_ids``, ``goal_position`` and ``alt_token``;
``alt_index`` and ``instruction`` are carried through onto the dataset row when present. The output
is the usual ``<prefix>_data.jsonl`` + ``<prefix>_index.sqlite`` pair, keyed by ``alt_token``.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import click

from route_description_generation.dataset_builder import build_dataset_from_routes
from route_description_generation.scenario_loading import (
    build_scenarios,
    resolve_data_path,
    resolve_maps_path,
)

REQUIRED_FIELDS = ("source_token", "route_roadblock_ids", "goal_position", "alt_token")


def read_route_rows(path: Path) -> List[Dict]:
    """Read and validate the alternative-route JSONL."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                raise click.ClickException(f"{path}:{line_no} is missing {', '.join(missing)}")
            rows.append(row)
    if not rows:
        raise click.ClickException(f"{path} contains no route rows")
    return rows


@click.command("build-dataset-from-routes")
@click.option(
    "--routes",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL of routes to describe (source_token, route_roadblock_ids, goal_position, alt_token).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("datasets"),
    show_default=True,
    help="Directory the dataset pair is written to.",
)
@click.option("--output-prefix", default="lg_routing", show_default=True, help="Dataset file prefix.")
@click.option("--split", default=None, help="Optional split name prepended to the dataset files.")
@click.option("--data-path", default=None, help="nuPlan database directory. Defaults to $NUPLAN_DATA_ROOT.")
@click.option("--map-path", default=None, help="nuPlan maps directory. Defaults to $NUPLAN_MAPS_ROOT.")
@click.option("--map-version", default="nuplan-maps-v1.0", show_default=True)
@click.option(
    "--extraction-offset",
    type=float,
    default=0.0,
    show_default=True,
    help=(
        "Scenario extraction offset [s]. Must match the pipeline that produced the routes: the ego "
        "pose the route was searched from is the pose the description starts at."
    ),
)
@click.option("--workers", type=int, default=12, show_default=True)
@click.option("--chunk-size", type=int, default=1000, show_default=True)
@click.option("--routing-service", default="osrm", show_default=True)
@click.option(
    "--route-goal-horizon-s",
    type=int,
    default=60,
    show_default=True,
    help="Recorded as routing_horizon_s. Keep it equal to the horizon the routes were searched to.",
)
def build_dataset_from_routes_command(
    routes: Path,
    output_dir: Path,
    output_prefix: str,
    split: Optional[str],
    data_path: Optional[str],
    map_path: Optional[str],
    map_version: str,
    extraction_offset: float,
    workers: int,
    chunk_size: int,
    routing_service: str,
    route_goal_horizon_s: int,
) -> None:
    """Generate route descriptions for externally supplied routes."""
    rows = read_route_rows(routes)

    by_source: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source_token"]].append(row)
    click.echo(f"{len(rows):,} routes across {len(by_source):,} scenarios")

    resolved_data_path = Path(data_path) if data_path else resolve_data_path(split)
    if resolved_data_path is None or not resolved_data_path.is_dir():
        raise click.ClickException(
            "Cannot determine the nuPlan database directory. Pass --data-path, or set "
            "NUPLAN_DATA_ROOT together with --split."
        )
    resolved_map_path = Path(map_path) if map_path else resolve_maps_path()
    if resolved_map_path is None or not resolved_map_path.is_dir():
        raise click.ClickException(
            "Cannot determine the nuPlan maps directory. Pass --map-path or set NUPLAN_MAPS_ROOT."
        )

    scenarios = build_scenarios(
        data_path=str(resolved_data_path),
        map_path=str(resolved_map_path),
        map_version=map_version,
        log_names=None,
        scenario_tokens=sorted(by_source),
        total_scenarios=None,
        scenarios_per_type=None,
        shuffle_scenarios=False,
        extraction_offset=extraction_offset,
    )
    click.echo(f"Loaded {len(scenarios):,} scenarios")

    # One scenario can carry several alternatives; each becomes its own dataset row.
    items = [
        (scenario, alternative)
        for scenario in scenarios
        for alternative in by_source.get(scenario.token, ())
    ]
    missing = len(rows) - len(items)
    if missing:
        click.echo(f"WARNING: {missing:,} routes had no loadable scenario and are skipped")

    build_dataset_from_routes(
        items,
        output_dir,
        split=split,
        output_prefix=output_prefix,
        workers=workers,
        chunk_size=chunk_size,
        routing_service=routing_service,
        route_goal_horizon_s=route_goal_horizon_s,
    )
