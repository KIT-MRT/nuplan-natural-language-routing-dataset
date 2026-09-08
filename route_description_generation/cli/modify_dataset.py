"""One-off post-processing of an already-generated dataset.

Rewrites a JSONL + SQLite dataset in place without re-running any routing. Edit
:func:`modify_scenario` to describe the transformation you want, then run ``rdg modify-dataset``.
"""

import json
import sqlite3
from pathlib import Path

import click

from route_description_generation.osrm_client import extract_maneuver_positions


def modify_scenario(scenario: dict) -> dict:
    """
    Modify a single JSONL scenario in-place WITHOUT re-running routing.

    Change this function to implement any logic you want.
    """
    routing_data = scenario.get("routing_data", {})
    route_directions = routing_data.get("route_directions", None)

    # Example: add position where maneuvers occur
    epsg = routing_data.get("route_epsg", None)
    if route_directions is not None:
        try:
            route_maneuver_positions = extract_maneuver_positions(
                route_directions, epsg, max_maneuvers=20
            )
        except (KeyError, IndexError, TypeError):
            route_maneuver_positions = []
            print(
                f"Warning: could not extract maneuver positions for scenario {scenario.get('token', 'unknown')}"
            )
        routing_data["route_maneuver_positions"] = route_maneuver_positions

    scenario["routing_data"] = routing_data
    return scenario


def rewrite_dataset(input_dir: Path, base_name: str = "lg_routing") -> int:
    """Apply :func:`modify_scenario` to every row of an existing dataset, in place.

    Writes to temporary files and swaps them in only once the whole pass succeeded, so an
    interrupted run cannot leave a half-rewritten dataset behind. No routing calls are made.
    """
    data_file = input_dir / f"{base_name}_data.jsonl"
    index_file = input_dir / f"{base_name}_index.sqlite"
    if not data_file.exists():
        raise FileNotFoundError(f"{data_file} not found")
    if not index_file.exists():
        raise FileNotFoundError(f"{index_file} not found")

    tmp_data = input_dir / f"{base_name}_data.tmp.jsonl"
    tmp_index = input_dir / f"{base_name}_index.tmp.sqlite"

    connection = sqlite3.connect(tmp_index)
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE routes (
            token TEXT PRIMARY KEY,
            offset INTEGER NOT NULL
        )
        """
    )
    connection.commit()

    processed = 0
    try:
        with open(data_file, "r") as source, open(tmp_data, "w", buffering=1) as target:
            for line in source:
                scenario = modify_scenario(json.loads(line))
                offset = target.tell()
                target.write(json.dumps(scenario) + "\n")
                cursor.execute(
                    "INSERT OR REPLACE INTO routes (token, offset) VALUES (?, ?)",
                    (scenario["token"], offset),
                )
                processed += 1
                if processed % 10_000 == 0:
                    connection.commit()
                    print(f"Processed {processed:,} scenarios")
        connection.commit()
    finally:
        connection.close()

    tmp_data.replace(data_file)
    tmp_index.replace(index_file)
    return processed


@click.command("modify-dataset")
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory holding the dataset pair to rewrite in place.",
)
@click.option("--base-name", default="lg_routing", show_default=True)
def modify_dataset_command(input_dir: Path, base_name: str) -> None:
    """Re-run `modify_scenario` over an existing dataset without any routing calls."""
    processed = rewrite_dataset(input_dir, base_name)
    click.echo(f"Done. Modified {processed:,} scenarios.")
