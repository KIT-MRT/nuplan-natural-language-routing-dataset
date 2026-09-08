"""``rdg build-dataset`` — generate a routing dataset from nuPlan scenarios."""

from pathlib import Path
from typing import Optional

import click

from route_description_generation.dataset_builder import build_dataset
from route_description_generation.scenario_loading import (
    load_token_filter,
    resolve_data_path,
    resolve_extraction_offset,
    resolve_log_names,
    resolve_maps_path,
    resolve_scenario_tokens_file,
)

_DEFAULT_SPLIT_LOGS_DIR = Path(__file__).resolve().parents[2] / "res"


@click.command("build-dataset")
@click.option(
    "--data-path",
    default=None,
    help=(
        "nuPlan database directory. Defaults to the one implied by --split: train/val/val14 use "
        "$NUPLAN_DATA_ROOT/nuplan-v1.1/splits/trainval, anything else uses .../test."
    ),
)
@click.option(
    "--map-path",
    default=None,
    help="nuPlan maps directory. Defaults to $NUPLAN_MAPS_ROOT.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("datasets"),
    show_default=True,
    help="Directory the dataset pair is written to, relative to the working directory.",
)
@click.option(
    "--split",
    type=click.Choice(["train", "val", "val14", "interplan"]),
    default=None,
    help="Shortcut for the split-specific log-name file in --split-logs-dir.",
)
@click.option(
    "--log-names-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Explicit log-name list, overriding --split.",
)
@click.option(
    "--split-logs-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=_DEFAULT_SPLIT_LOGS_DIR,
    show_default=True,
    help="Directory holding the split log-name json files.",
)
@click.option(
    "--scenario-tokens-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Restrict to these tokens, one per line (see `rdg export-tokens`). Defaults to "
        "res/<split>_tokens.txt when that file ships for the split."
    ),
)
@click.option("--output-prefix", default="lg_routing", show_default=True)
@click.option("--map-version", default="nuplan-maps-v1.0", show_default=True)
@click.option("--workers", type=int, default=12, show_default=True)
@click.option(
    "--chunk-size",
    type=int,
    default=1000,
    show_default=True,
    help="Scenarios in flight at once; bounds peak memory.",
)
@click.option("--total-scenarios", type=int, default=None)
@click.option("--scenarios-per-type", type=int, default=None)
@click.option("--shuffle-scenarios", is_flag=True)
@click.option(
    "--enable-interplan-variants",
    is_flag=True,
    help="Also route left/right/straight interplan goal variants.",
)
@click.option("--route-goal-horizon-s", type=int, default=60, show_default=True)
@click.option("--sample-frequency-hz", type=int, default=10, show_default=True)
@click.option(
    "--extraction-offset",
    type=float,
    default=None,
    help=(
        "Seconds of run-up loaded before the tagged token, so the route starts where a simulated "
        "planner does. Defaults to -3.0 for val/val14/interplan (matching nuPlan's simulation "
        "builders) and 0.0 for train."
    ),
)
def build_dataset_command(
    data_path: Optional[str],
    map_path: Optional[str],
    output_dir: Path,
    split: Optional[str],
    log_names_json: Optional[Path],
    split_logs_dir: Path,
    scenario_tokens_file: Optional[Path],
    output_prefix: str,
    map_version: str,
    workers: int,
    chunk_size: int,
    total_scenarios: Optional[int],
    scenarios_per_type: Optional[int],
    shuffle_scenarios: bool,
    enable_interplan_variants: bool,
    route_goal_horizon_s: int,
    sample_frequency_hz: int,
    extraction_offset: Optional[float],
) -> None:
    """Build a natural-language routing dataset from nuPlan scenarios.

    Requires the OSRM servers for the relevant map regions to be running; see
    `scripts/start_osrm_servers.sh`.
    """
    resolved_data_path = Path(data_path) if data_path else resolve_data_path(split)
    if resolved_data_path is None:
        raise click.ClickException(
            "Cannot determine the nuPlan database directory. Pass --data-path, or set "
            "NUPLAN_DATA_ROOT together with --split."
        )
    if not resolved_data_path.is_dir():
        raise click.ClickException(f"nuPlan database directory not found: {resolved_data_path}")

    resolved_map_path = Path(map_path) if map_path else resolve_maps_path()
    if resolved_map_path is None:
        raise click.ClickException(
            "Cannot determine the nuPlan maps directory. Pass --map-path or set NUPLAN_MAPS_ROOT."
        )
    if not resolved_map_path.is_dir():
        raise click.ClickException(f"nuPlan maps directory not found: {resolved_map_path}")

    if not data_path or not map_path:
        click.echo(f"Using data path {resolved_data_path}")
        click.echo(f"Using map path  {resolved_map_path}")

    resolved_extraction_offset = (
        extraction_offset if extraction_offset is not None else resolve_extraction_offset(split)
    )
    if resolved_extraction_offset != 0.0:
        click.echo(
            f"Using extraction offset {resolved_extraction_offset}s: routes start where the "
            "simulation does, not at the tagged token"
        )

    log_names = resolve_log_names(
        split=split, log_names_json=log_names_json, split_logs_dir=split_logs_dir
    )

    # Some splits are defined by their token list, not by log names; see
    # resolve_scenario_tokens_file. An explicit --scenario-tokens-file always wins.
    tokens_file = scenario_tokens_file or resolve_scenario_tokens_file(split, split_logs_dir)
    scenario_tokens = None
    if tokens_file is not None:
        scenario_tokens = load_token_filter(tokens_file)
        click.echo(f"Loaded {len(scenario_tokens):,} scenario tokens from {tokens_file}")
    else:
        # Say so explicitly. A split whose token list has gone missing would otherwise widen
        # silently - val14 without res/val14_tokens.txt is the whole val split, not 1118 scenarios.
        click.echo(
            f"No scenario token filter for split {split!r} "
            f"(no {split_logs_dir / f'{split}_tokens.txt'}); using every scenario in the split"
            if split
            else "No scenario token filter; using every scenario matched by the log-name filter"
        )

    build_dataset(
        data_path=str(resolved_data_path),
        map_path=str(resolved_map_path),
        output_dir=output_dir,
        split=split,
        output_prefix=output_prefix,
        map_version=map_version,
        log_names=log_names,
        scenario_tokens=scenario_tokens,
        total_scenarios=total_scenarios,
        scenarios_per_type=scenarios_per_type,
        shuffle_scenarios=shuffle_scenarios,
        enable_interplan_variants=enable_interplan_variants,
        workers=workers,
        chunk_size=chunk_size,
        route_goal_horizon_s=route_goal_horizon_s,
        sample_frequency_hz=sample_frequency_hz,
        extraction_offset=resolved_extraction_offset,
    )
