#!/usr/bin/env python3
"""Create a nuPlan ScenarioFilter YAML from JSONL rows with routing_data.valid_route=true.

Usage:
  python scripts/create_valid_route_scenario_filter.py \
      --input-jsonl datasets/lg_routing_data.jsonl \
    [--output-yaml /custom/output/path.yaml]

Optional:
  --template-yaml flow_drive_planner/flow_drive/config/scenario_filter/val14.yaml

If --template-yaml is provided, all template keys are preserved and only
"scenario_tokens" is replaced.

Default output path when --output-yaml is omitted:
- same folder as input JSONL
- name is first underscore-delimited part of input stem + ".yaml"
    Example: val14_lg_data.jsonl -> val14.yaml
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

from route_description_generation.dataset_index import collect_tokens


class _DoubleQuotedToken(str):
    """Marker subtype used to force double-quoted YAML output for scenario tokens."""


class _ScenarioFilterDumper(yaml.SafeDumper):
    """YAML dumper that supports per-type scalar style overrides."""


def _represent_double_quoted_token(dumper: yaml.SafeDumper, data: _DoubleQuotedToken):
    return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style='"')


_ScenarioFilterDumper.add_representer(
    _DoubleQuotedToken,
    _represent_double_quoted_token,
)


def _default_filter_yaml(valid_tokens: List[str]) -> Dict[str, Any]:
    return {
        "_target_": "nuplan.planning.scenario_builder.scenario_filter.ScenarioFilter",
        "_convert_": "all",
        "scenario_types": None,
        "scenario_tokens": valid_tokens,
        "log_names": "${splitter.log_splits.val}",
        "map_names": None,
        "num_scenarios_per_type": 100,
        "limit_total_scenarios": None,
        "timestamp_threshold_s": 15,
        "ego_displacement_minimum_m": None,
        "ego_start_speed_threshold": None,
        "ego_stop_speed_threshold": None,
        "speed_noise_tolerance": None,
        "expand_scenarios": False,
        "remove_invalid_goals": True,
        "shuffle": False,
    }


def _build_output_yaml(template_yaml: Path | None, valid_tokens: List[str]) -> Dict[str, Any]:
    if template_yaml is None:
        return _default_filter_yaml(valid_tokens)

    with template_yaml.open("r", encoding="utf-8") as f:
        template = yaml.safe_load(f) or {}

    if not isinstance(template, dict):
        raise ValueError(f"Template YAML must be a mapping: {template_yaml}")

    template["scenario_tokens"] = valid_tokens
    return template


def _quote_yaml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _write_yaml_with_template_format(
    template_yaml: Path,
    output_yaml: Path,
    valid_tokens: List[str],
) -> None:
    """Replace only the scenario_tokens block in template text, preserving formatting."""
    text = template_yaml.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    key_idx = -1
    key_indent = ""
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)scenario_tokens:\s*(?:#.*)?$", line.rstrip("\n"))
        if m:
            key_idx = i
            key_indent = m.group(1)
            break

    if key_idx < 0:
        raise ValueError(f"Template YAML has no 'scenario_tokens:' key: {template_yaml}")

    key_indent_len = len(key_indent)

    item_indent = key_indent + "  "
    for j in range(key_idx + 1, len(lines)):
        m_item = re.match(r"^(\s*)-\s+", lines[j])
        if m_item and len(m_item.group(1)) > key_indent_len:
            item_indent = m_item.group(1)
            break

    end_idx = len(lines)
    for j in range(key_idx + 1, len(lines)):
        raw = lines[j]
        stripped = raw.strip()
        if not stripped:
            continue
        cur_indent = len(raw) - len(raw.lstrip(" "))
        if cur_indent <= key_indent_len:
            end_idx = j
            break

    token_lines = [f"{item_indent}- {_quote_yaml_string(token)}\n" for token in valid_tokens]
    new_lines = lines[: key_idx + 1] + token_lines + lines[end_idx:]

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    output_yaml.write_text("".join(new_lines), encoding="utf-8")


def _default_output_yaml_path(input_jsonl: Path) -> Path:
    stem = input_jsonl.stem
    prefix = stem.split("_", 1)[0] if "_" in stem else stem
    if not prefix:
        raise ValueError(f"Cannot derive output name from input file: {input_jsonl}")
    return input_jsonl.parent / f"{prefix}.yaml"


def _force_double_quoted_tokens(doc: Dict[str, Any]) -> Dict[str, Any]:
    tokens = doc.get("scenario_tokens")
    if isinstance(tokens, list):
        doc["scenario_tokens"] = [
            _DoubleQuotedToken(t) if isinstance(t, str) else t for t in tokens
        ]
    return doc


@click.command("scenario-filter")
@click.option(
    "--input-jsonl",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Generated dataset JSONL to read valid tokens from.",
)
@click.option(
    "--output-yaml",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where to write the filter. Defaults next to the input JSONL.",
)
@click.option(
    "--template-yaml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Existing ScenarioFilter to update; only scenario_tokens is replaced.",
)
def scenario_filter_command(
    input_jsonl: Path, output_yaml: Optional[Path], template_yaml: Optional[Path]
) -> None:
    """Emit a nuPlan ScenarioFilter YAML holding only the dataset's valid routes."""
    destination = output_yaml or _default_output_yaml_path(input_jsonl)
    valid_tokens = collect_tokens(input_jsonl, valid_route=True)

    if template_yaml is not None:
        _write_yaml_with_template_format(template_yaml, destination, valid_tokens)
    else:
        document = _force_double_quoted_tokens(_build_output_yaml(None, valid_tokens))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as out:
            yaml.dump(
                document,
                out,
                Dumper=_ScenarioFilterDumper,
                sort_keys=False,
                default_flow_style=False,
            )

    click.echo(f"Wrote {len(valid_tokens)} valid token(s) to {destination}")
