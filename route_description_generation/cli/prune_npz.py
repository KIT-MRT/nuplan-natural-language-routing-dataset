"""``rdg prune-npz`` — reduce a directory of NPZ files to a trainable set.

Two passes, both moving files into subdirectories rather than deleting anything:

1. scenarios the routing dataset marks invalid go to ``invalid/``;
2. if more than ``--num-scenarios`` remain, a random excess goes to ``surplus/``.

Any ``scenarios_*.json`` list the data processor wrote alongside the NPZ files is rewritten to match,
with the original kept as a ``.bak``.

The usual order of operations is to export tokens from the NPZ directory, build a routing dataset
from them, then run this so what stays at the top level is exactly the set to train on.
"""

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set

import click
from tqdm import tqdm

from route_description_generation.cli.export_tokens import token_from_npz_filename
from route_description_generation.dataset_index import collect_tokens

# Subdirectories created inside the NPZ directory. Nothing is ever deleted, so a run can be undone
# by moving the files back.
INVALID_SUBDIR = "invalid"
SURPLUS_SUBDIR = "surplus"

# Scenario lists written next to the NPZ files by flow_drive_planner's data_process.py (one flat
# list of .npz filenames, named per split). Training reads these rather than the directory, so a
# pruned directory with an unpruned list would still load the files that were moved aside.
SCENARIO_LIST_GLOB = "scenarios_*.json"
BACKUP_SUFFIX = ".bak"


class ScenarioListUpdate(NamedTuple):
    path: Path
    before: int
    after: int
    backup: Optional[Path]


class PruneSummary(NamedTuple):
    moved_invalid: int
    moved_surplus: int
    kept: int
    missing: int
    blocked: List[str]
    scenario_lists: List[ScenarioListUpdate]


def index_npz_by_token(npz_dir: Path, *, show_progress: bool = False) -> Dict[str, str]:
    """Map scenario token -> NPZ *filename* for ``<map_name>_<token>.npz`` files in ``npz_dir``.

    Only the top level is scanned, so files already moved into a subdirectory are left out and
    re-running is harmless.
    """
    by_token: Dict[str, str] = {}
    entries = tqdm(
        os.scandir(npz_dir),
        desc="Indexing NPZ files",
        unit="file",
        disable=not show_progress,
    )
    for entry in entries:
        token = token_from_npz_filename(entry.name)
        if token:
            by_token[token] = entry.name
    return by_token


def _existing_names(directory: Path) -> Set[str]:
    """Filenames already in ``directory``, or an empty set when it does not exist.

    Read once so the no-clobber check below is a set lookup rather than a stat per file.
    """
    try:
        with os.scandir(directory) as entries:
            return {entry.name for entry in entries}
    except FileNotFoundError:
        return set()


def _move_all(
    filenames: List[str],
    source_dir: Path,
    destination_dir: Path,
    *,
    dry_run: bool,
    show_progress: bool = False,
) -> tuple:
    """Move ``filenames`` from ``source_dir`` into ``destination_dir``.

    Returns ``(moved_count, blocked_names)``.
    """
    if not filenames:
        return 0, []
    occupied = _existing_names(destination_dir)
    if not dry_run:
        destination_dir.mkdir(exist_ok=True)

    moved = 0
    blocked: List[str] = []
    for name in tqdm(
        filenames,
        desc=f"Moving to {destination_dir.name}/",
        unit="file",
        disable=not show_progress,
    ):
        if name in occupied:
            # Never clobber a previous run's file; report it instead.
            blocked.append(name)
            continue
        if not dry_run:
            source = source_dir / name
            destination = destination_dir / name
            try:
                # A plain rename within the same filesystem, which a subdirectory almost always
                # is. shutil.move would stat the destination first and is only needed when the
                # subdirectory turns out to be a separate mount.
                os.rename(source, destination)
            except OSError:
                shutil.move(str(source), str(destination))
        moved += 1
    return moved, blocked


def find_scenario_lists(npz_dir: Path) -> List[Path]:
    """Scenario-list JSONs sitting next to the NPZ files. Backups end in ``.bak`` and never match."""
    return sorted(npz_dir.glob(SCENARIO_LIST_GLOB))


def rewrite_scenario_list(
    path: Path, moved: Set[str], *, dry_run: bool = False
) -> ScenarioListUpdate:
    """Drop ``moved`` filenames from a scenario list, keeping the original as a ``.bak``.

    Entries are *subtracted* rather than the list being rebuilt from the directory, so anything the
    NPZ filename parser does not recognise survives untouched. The backup is written only once: on a
    second run the existing ``.bak`` is the pre-prune original and must not be replaced by an
    already-pruned list.
    """
    with path.open("r", encoding="utf-8") as file:
        names = json.load(file)

    kept = [name for name in names if name not in moved]
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    if len(kept) == len(names):
        return ScenarioListUpdate(path=path, before=len(names), after=len(kept), backup=None)

    if not dry_run:
        if not backup.exists():
            shutil.copy2(path, backup)
        # Write beside the target and rename, so an interrupted run cannot leave a half-written
        # list where training expects a complete one.
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(kept, file, indent=4)
        os.replace(temporary, path)

    return ScenarioListUpdate(
        path=path, before=len(names), after=len(kept), backup=backup if not dry_run else None
    )


def update_scenario_lists(
    npz_dir: Path,
    moved: Set[str],
    *,
    scenario_lists: Optional[Sequence[Path]] = None,
    dry_run: bool = False,
) -> List[ScenarioListUpdate]:
    paths = list(scenario_lists) if scenario_lists is not None else find_scenario_lists(npz_dir)
    return [rewrite_scenario_list(path, moved, dry_run=dry_run) for path in paths]


def prune_npz(
    npz_dir: Path,
    dataset_file: Path,
    *,
    num_scenarios: Optional[int] = None,
    seed: int = 0,
    dry_run: bool = False,
    show_progress: bool = False,
    update_lists: bool = True,
) -> PruneSummary:
    """Quarantine invalid scenarios, then cap what remains at ``num_scenarios``."""
    by_token = index_npz_by_token(npz_dir, show_progress=show_progress)
    invalid_tokens = collect_tokens(dataset_file, valid_route=False, show_progress=show_progress)

    invalid_names = [by_token[t] for t in invalid_tokens if t in by_token]
    missing = len(invalid_tokens) - len(invalid_names)
    moved_invalid, blocked = _move_all(
        invalid_names,
        npz_dir,
        npz_dir / INVALID_SUBDIR,
        dry_run=dry_run,
        show_progress=show_progress,
    )

    # What is left at the top level once the invalid ones are gone. Computed rather than re-listed
    # so the count is right under --dry-run too, where nothing has actually moved.
    quarantined = set(invalid_names) - set(blocked)
    remaining = sorted(name for name in by_token.values() if name not in quarantined)

    moved_surplus = 0
    moved = set(quarantined)
    if num_scenarios is not None and len(remaining) > num_scenarios:
        # Sorted before sampling so the seed alone decides the selection - directory listing
        # order is arbitrary and would otherwise make runs irreproducible.
        surplus = random.Random(seed).sample(remaining, len(remaining) - num_scenarios)
        moved_surplus, surplus_blocked = _move_all(
            surplus,
            npz_dir,
            npz_dir / SURPLUS_SUBDIR,
            dry_run=dry_run,
            show_progress=show_progress,
        )
        blocked = blocked + surplus_blocked
        moved |= set(surplus) - set(surplus_blocked)

    # Training loads the scenario list, not the directory, so leaving it unpruned would keep
    # loading the files that were just moved aside.
    scenario_lists = update_scenario_lists(npz_dir, moved, dry_run=dry_run) if update_lists else []

    return PruneSummary(
        moved_invalid=moved_invalid,
        moved_surplus=moved_surplus,
        kept=len(remaining) - moved_surplus,
        missing=missing,
        blocked=blocked,
        scenario_lists=scenario_lists,
    )


@click.command("prune-npz")
@click.option(
    "--npz-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of <map_name>_<token>.npz files to prune.",
)
@click.option(
    "--dataset",
    "dataset_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Routing dataset JSONL whose valid_route flags decide what is invalid.",
)
@click.option(
    "--num-scenarios",
    type=int,
    default=None,
    help=(
        "Cap the kept set at this many scenarios. A random excess is moved to "
        f"{SURPLUS_SUBDIR}/. Omit to keep every valid scenario."
    ),
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed for the random excess selection, so a run can be reproduced.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report what would move without touching anything.",
)
@click.option(
    "--update-lists/--no-update-lists",
    default=True,
    show_default=True,
    help=(
        f"Rewrite any {SCENARIO_LIST_GLOB} next to the NPZ files to drop the moved scenarios, "
        f"keeping the original as *{BACKUP_SUFFIX}."
    ),
)
@click.option("--progress/--no-progress", default=True, show_default=True)
def prune_npz_command(
    npz_dir: Path,
    dataset_file: Path,
    num_scenarios: Optional[int],
    seed: int,
    dry_run: bool,
    update_lists: bool,
    progress: bool,
) -> None:
    """Move invalid NPZ files aside, then cap the remainder at --num-scenarios."""
    summary = prune_npz(
        npz_dir,
        dataset_file,
        num_scenarios=num_scenarios,
        seed=seed,
        dry_run=dry_run,
        show_progress=progress,
        update_lists=update_lists,
    )

    verb = "Would move" if dry_run else "Moved"
    click.echo(f"{verb} {summary.moved_invalid:,} invalid NPZ files to {INVALID_SUBDIR}/")
    if num_scenarios is not None:
        click.echo(
            f"{verb} {summary.moved_surplus:,} surplus NPZ files to {SURPLUS_SUBDIR}/ (seed {seed})"
        )
    click.echo(f"{summary.kept:,} scenarios remain in {npz_dir}")

    for update in summary.scenario_lists:
        if update.before == update.after:
            click.echo(f"{update.path.name}: already matches ({update.after:,} entries)")
            continue
        verb = "Would rewrite" if dry_run else "Rewrote"
        backup = f", original kept as {update.backup.name}" if update.backup else ""
        click.echo(
            f"{verb} {update.path.name}: {update.before:,} -> {update.after:,} entries{backup}"
        )
    if update_lists and not summary.scenario_lists:
        click.echo(f"No {SCENARIO_LIST_GLOB} found in {npz_dir}; nothing to rewrite")

    if summary.missing:
        click.echo(
            f"{summary.missing:,} invalid scenarios have no NPZ file here (already moved, "
            "or the dataset covers a different directory)"
        )
    if summary.blocked:
        click.echo(
            f"WARNING: {len(summary.blocked):,} left in place - a file of the same name is "
            f"already in the target directory (e.g. {summary.blocked[0]})"
        )
