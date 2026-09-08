"""Command-line entry point for route_description_generation."""

import click

from route_description_generation.cli.build_dataset import build_dataset_command
from route_description_generation.cli.build_dataset_from_routes import (
    build_dataset_from_routes_command,
)
from route_description_generation.cli.export_tokens import export_tokens_command
from route_description_generation.cli.modify_dataset import modify_dataset_command
from route_description_generation.cli.prune_npz import prune_npz_command
from route_description_generation.cli.scenario_filter import scenario_filter_command


@click.group()
@click.version_option(package_name="route_description_generation")
def cli() -> None:
    """Build natural-language driving-route datasets for nuPlan scenarios."""


cli.add_command(build_dataset_command)
cli.add_command(build_dataset_from_routes_command)
cli.add_command(export_tokens_command)
cli.add_command(modify_dataset_command)
cli.add_command(prune_npz_command)
cli.add_command(scenario_filter_command)
