"""``rdg export-tokens`` — collect scenario tokens from a directory of NPZ files."""

import os
from pathlib import Path
from typing import Optional

import click
from tqdm import tqdm

_DEFAULT_RES_DIR = Path(__file__).resolve().parents[2] / "res"


def token_from_npz_filename(filename: str):
    """Token from a ``<map_name>_<token>.npz`` filename, or ``None`` if it does not match."""
    if not filename.endswith(".npz"):
        return None

    stem = filename[:-4]
    if "_" not in stem:
        return None

    return stem.rsplit("_", 1)[-1]


def export_tokens(
    input_dir: Path,
    output_file: Path,
    flush_every: int = 10000,
    show_progress: bool = True,
) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    buffer = []

    # Stream directory entries to avoid materializing very large file lists in memory.
    with output_file.open("w", encoding="utf-8") as out:
        with os.scandir(input_dir) as entries:
            iterator = entries
            if show_progress:
                iterator = tqdm(entries, desc="Scanning NPZ files", unit="files")

            for entry in iterator:
                if not entry.is_file():
                    continue
                token = token_from_npz_filename(entry.name)
                if token is None:
                    continue

                buffer.append(token)
                count += 1

                if len(buffer) >= flush_every:
                    out.write("\n".join(buffer) + "\n")
                    buffer.clear()

        if buffer:
            out.write("\n".join(buffer) + "\n")

    return count


@click.command("export-tokens")
@click.option(
    "--split",
    type=click.Choice(["train", "val", "val14", "interplan"]),
    default="train",
    show_default=True,
    help="Split used to derive the default input and output paths.",
)
@click.option(
    "--npz-dataset-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    required=True,
    help="Root directory containing the per-split NPZ subfolders.",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Token file to write. Defaults to res/<split>_tokens.txt.",
)
@click.option("--flush-every", type=int, default=10000, show_default=True)
@click.option("--progress/--no-progress", default=True, show_default=True)
def export_tokens_command(
    split: str,
    npz_dataset_root: Path,
    output_file: Optional[Path],
    flush_every: int,
    progress: bool,
) -> None:
    """Export scenario tokens from a directory of `<map>_<token>.npz` files."""
    input_dir = npz_dataset_root / split
    if not input_dir.is_dir():
        raise click.ClickException(f"Not a directory: {input_dir}")

    destination = output_file or (_DEFAULT_RES_DIR / f"{split}_tokens.txt")
    exported = export_tokens(
        input_dir, destination, flush_every=flush_every, show_progress=progress
    )
    click.echo(f"Exported {exported:,} tokens to {destination}")
