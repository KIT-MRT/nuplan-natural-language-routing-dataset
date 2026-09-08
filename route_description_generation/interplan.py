from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import yaml

InterplanGoals = Dict[str, Optional[Tuple[float, float]]]


def _parse_goal_tuple(value: Optional[str]) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    return (float(parts[0]), float(parts[1]))


def load_interplan_modification_goals(path: Path) -> Dict[str, InterplanGoals]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    mod_details = data.get("modification_details_dictionary", {})
    goals_by_token: Dict[str, InterplanGoals] = {}
    for token, details in mod_details.items():
        goal_data = (details or {}).get("goal", {})
        goals_by_token[str(token)] = {
            "left": _parse_goal_tuple(goal_data.get("left")),
            "right": _parse_goal_tuple(goal_data.get("right")),
            "straight": _parse_goal_tuple(goal_data.get("straight")),
        }
    return goals_by_token


def load_interplan_tokens(
    benchmark_yaml: Path,
    modifications_yaml: Path,
) -> List[str]:
    with benchmark_yaml.open("r", encoding="utf-8") as file:
        benchmark_data = yaml.safe_load(file) or {}
    benchmark_tokens_raw = benchmark_data.get("scenario_tokens", []) or []
    benchmark_tokens = {str(token).split("-")[0] for token in benchmark_tokens_raw}

    modifications_goals = load_interplan_modification_goals(modifications_yaml)
    interplan_tokens = benchmark_tokens.union(modifications_goals.keys())
    return cast(List[str], sorted(interplan_tokens))


def resolve_interplan_yaml_paths(
    benchmark_yaml: Optional[Path],
    modifications_yaml: Optional[Path],
) -> Tuple[Optional[Path], Optional[Path]]:
    if benchmark_yaml is not None and modifications_yaml is not None:
        return benchmark_yaml, modifications_yaml

    try:
        import interplan
    except ImportError:
        return benchmark_yaml, modifications_yaml

    interplan_root = Path(interplan.__path__[0])
    default_benchmark = (
        interplan_root
        / "planning"
        / "script"
        / "config"
        / "common"
        / "scenario_filter"
        / "benchmark_scenarios.yaml"
    )
    default_modifications = (
        interplan_root
        / "planning"
        / "script"
        / "config"
        / "common"
        / "scenario_filter"
        / "modifications"
        / "interPlan_modifications.yaml"
    )

    resolved_benchmark = benchmark_yaml or (
        default_benchmark if default_benchmark.exists() else None
    )
    resolved_modifications = modifications_yaml or (
        default_modifications if default_modifications.exists() else None
    )
    return resolved_benchmark, resolved_modifications
