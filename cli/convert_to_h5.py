"""
Convert particle images with poses and CTF into HDF5 format.

Combines particle images (.star, .cs, .mrc, .mrcs, .txt) with
poses.pkl and ctf.pkl into a single HDF5 file for CryoUNI training.

HDF5 layout:
  - particles:           (N, D, D) float16
  - poses/R:             (N, 3, 3) float64
  - poses/T:             (N, 2) float64
  - ctf/imgsize_pix:     (N,) int32
  - ctf/psize_A:         (N,) float64
  - ctf/df1_A:           (N,) float64
  - ctf/df2_A:           (N,) float64
  - ctf/dfang_deg:       (N,) float64
  - ctf/vol_kv:          (N,) float64
  - ctf/cs_mm:           (N,) float64
  - ctf/w:               (N,) float64
  - ctf/phase_shift_deg: (N,) float64
"""

from __future__ import annotations

from pathlib import Path
import pickle

import h5py
import numpy as np
import typer
from typing_extensions import Annotated

from hetero_recon.utils.io import read_cryodrgn_particle

app = typer.Typer(
    help="Convert particles + poses + CTF pickles into CryoUNI HDF5 format.",
    add_completion=False,
)


@app.command()
def main(
    particle_path: Annotated[Path, typer.Option("--particles", "-i", help="Particle images (.star, .cs, .mrc, .mrcs, .txt)", exists=True)],
    poses_path: Annotated[Path, typer.Option("--poses", "-p", help="Poses pickle file", exists=True)],
    ctf_path: Annotated[Path, typer.Option("--ctf", "-c", help="CTF pickle file", exists=True)],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="Output HDF5 file path")],
    datadir: Annotated[Path | None,
                       typer.Option("--datadir", help="Directory containing particle .mrcs stacks (for .star/.cs inputs)")] = None,
    chunk_size: Annotated[int, typer.Option("--chunk-size", help="Number of particles to write per batch")] = 1000,
) -> None:
    """Convert particle images + poses + CTF into a single HDF5 file."""

    if out_path.exists():
        print(f"Error: output file already exists: {out_path}")
        print("  Delete it first or specify a different path.")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Load poses
    # ------------------------------------------------------------------
    print(f"Loading poses from {poses_path}")
    with open(poses_path, "rb") as f:
        rotations, translations = pickle.load(f)
    rotations = np.asarray(rotations, dtype=np.float64)
    translations = np.asarray(translations, dtype=np.float64)
    N = len(rotations)
    print(f"  {N} particles, rotations: {rotations.shape}, translations: {translations.shape}")

    # ------------------------------------------------------------------
    # Load CTF
    # ------------------------------------------------------------------
    print(f"Loading CTF from {ctf_path}")
    with open(ctf_path, "rb") as f:
        ctf = pickle.load(f)
    ctf = np.asarray(ctf, dtype=np.float64)
    if len(ctf) != N:
        print(f"Error: CTF count ({len(ctf)}) != poses count ({N})")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Load particles (lazy to avoid loading all into memory)
    # ------------------------------------------------------------------
    print(f"Loading particles from {particle_path}")
    particle_dir = str(datadir) if datadir is not None else None
    particles = read_cryodrgn_particle(particle_path, particle_dir, lazy=True)
    if len(particles) != N:
        print(f"Error: particle count ({len(particles)}) != poses count ({N})")
        raise typer.Exit(code=1)

    # Get image size from first particle
    first_img = particles[0].get() if hasattr(particles[0], "get") else particles[0]
    D = first_img.shape[-1]
    print(f"  Image size D={D}")

    # ------------------------------------------------------------------
    # Write HDF5
    # ------------------------------------------------------------------
    print(f"Writing HDF5 to {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        # Particle images — write in chunks to limit memory usage
        particles_ds = f.create_dataset(
            "particles",
            shape=(N, D, D),
            dtype=np.float16,
            chunks=(1, D, D),
        )
        is_lazy = hasattr(particles[0], "get")
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            if is_lazy:
                batch = np.stack([particles[i].get() for i in range(start, end)])
            else:
                batch = particles[start : end]
            particles_ds[start : end] = batch.astype(np.float16)
            print(f"  [{end}/{N}] particles written", end="\r")
        print()

        # Poses
        f.create_dataset("poses/R", data=rotations)
        f.create_dataset("poses/T", data=translations)

        # CTF (split (N, 9) array into individual per-field datasets)
        f.create_dataset("ctf/imgsize_pix", data=ctf[:, 0].astype(np.int32))
        f.create_dataset("ctf/psize_A", data=ctf[:, 1])
        f.create_dataset("ctf/df1_A", data=ctf[:, 2])
        f.create_dataset("ctf/df2_A", data=ctf[:, 3])
        f.create_dataset("ctf/dfang_deg", data=ctf[:, 4])
        f.create_dataset("ctf/vol_kv", data=ctf[:, 5])
        f.create_dataset("ctf/cs_mm", data=ctf[:, 6])
        f.create_dataset("ctf/w", data=ctf[:, 7])
        f.create_dataset("ctf/phase_shift_deg", data=ctf[:, 8])

    print(f"Done. {out_path} ({N} particles, D={D})")


if __name__ == "__main__":
    app()
