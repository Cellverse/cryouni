"""
Convert pickle index files to RELION star files.

Reads a RELION star file and cluster indices (as pickle files) to generate
per-cluster star files for downstream processing.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import starfile
import typer
from typing_extensions import Annotated

app = typer.Typer()


@app.command()
def main(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output directory containing index files",
            exists=True,
            resolve_path=True,
        ),
    ],
    star_path: Annotated[Path, typer.Option("--star", "-s", help="Input star file", exists=True)],
) -> None:
    """
    Convert cluster index pickle files to RELION star format.

    For each `cluster_*.pkl` index file, generates a corresponding star file
    containing particles belonging to that cluster.
    """
    # Read star file
    print(f"Reading {star_path}")
    data_all = starfile.read(star_path)
    data_particles = data_all["particles"]
    print(f"Found {len(data_particles)} particles")

    # Find all index files
    index_files = list(output_dir.glob("cluster_*.pkl"))
    print(f"Found {len(index_files)} index files")
    for index_file in index_files:
        # Read index file
        with open(index_file, "rb") as f:
            index = pickle.load(f)

        # Write star file
        output_file = output_dir / index_file.name.replace(".pkl", ".star")
        print(f"Writing {output_file.name}, {len(index) / len(data_particles) * 100:.2f}% of particles")

        belonging_particles = data_particles.iloc[index].copy()
        belonging_particles.reset_index(drop=True, inplace=True)

        starfile.write(
            {
                "optics": data_all["optics"],
                "particles": belonging_particles,
            },
            output_file,
        )

    print("Done")


if __name__ == "__main__":
    app()
