"""
Parse poses and CTF parameters from RELION .star files.

Converts RELION star files into the pickle format expected by CryoUNI,
eliminating the need to install cryoDRGN for data preprocessing.

Output format is compatible with cryoDRGN's pickle conventions:
  - poses.pkl: tuple (rotations [N,3,3], translations [N,2])
  - ctf.pkl: numpy array [N,9] with columns:
    [D, Apix, DefocusU, DefocusV, DefocusAngle, Voltage, Cs, AmpContrast, PhaseShift]
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import typer
from typing_extensions import Annotated

app = typer.Typer(
    help="Parse poses and CTF from RELION .star files into CryoUNI pickle format.",
    add_completion=False,
)

# ============================================================================
# Star file parser (supports RELION 3.0 and 3.1)
# ============================================================================


def _parse_star_file(star_path: Path) -> dict[str, dict]:
    """
    Parse a RELION star file into a dict of data blocks.

    Returns:
        Dict mapping block name -> {"headers": [...], "data": np.ndarray of strings}.
        For RELION 3.1, there will be a "data_optics" block and a "data_particles" block.
        For RELION 3.0, there will be a single "data_" block.
    """
    blocks = {}
    current_block = None
    headers = []
    rows = []

    with open(star_path, "r") as f:
        for line in f:
            line = line.strip()

            # Detect block start
            if line.startswith("data_"):
                # Save previous block if any
                if current_block is not None and headers:
                    blocks[current_block] = {
                        "headers": headers,
                        "data": np.array(rows) if rows else np.empty((0, len(headers))),
                    }
                current_block = line
                headers = []
                rows = []
                continue

            if line == "loop_":
                headers = []
                rows = []
                continue

            # Collect headers
            if line.startswith("_"):
                headers.append(line.split()[0])
                continue

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Data row
            if headers and current_block is not None:
                parts = line.split()
                if len(parts) == len(headers):
                    rows.append(parts)

    # Save last block
    if current_block is not None and headers:
        blocks[current_block] = {
            "headers": headers,
            "data": np.array(rows) if rows else np.empty((0, len(headers))),
        }

    return blocks


def _get_column(block: dict, name: str) -> np.ndarray | None:
    """Get a column from a star file data block by header name."""
    if name in block["headers"]:
        idx = block["headers"].index(name)
        return block["data"][:, idx]
    return None


def _get_field_values(
    blocks: dict[str, dict],
    field: str,
    n_particles: int,
) -> np.ndarray | None:
    """
    Get per-particle values for a field, handling RELION 3.0 and 3.1.

    In 3.1, some fields live in the optics table and must be mapped
    per-particle via _rlnOpticsGroup.
    """
    # Identify the main data block
    main_block = blocks.get("data_particles") or blocks.get("data_")
    if main_block is None:
        raise ValueError("No data block found in star file.")

    # Try main data block first
    col = _get_column(main_block, field)
    if col is not None:
        return col.astype(np.float64)

    # Try optics table (RELION 3.1)
    optics_block = blocks.get("data_optics")
    if optics_block is not None:
        col = _get_column(optics_block, field)
        if col is not None:
            # Map per-particle via _rlnOpticsGroup
            particle_groups = _get_column(main_block, "_rlnOpticsGroup")
            optics_groups = _get_column(optics_block, "_rlnOpticsGroup")
            if particle_groups is not None and optics_groups is not None:
                group_to_idx = {
                    g: i
                    for i, g in enumerate(optics_groups)
                }
                values = np.array([float(col[group_to_idx[pg]]) for pg in particle_groups])
                return values

    return None


# ============================================================================
# Rotation conversion (RELION Euler angles -> rotation matrix)
# ============================================================================


def _R_from_relion(euler: np.ndarray) -> np.ndarray:
    """
    Convert RELION Euler angles (rot, tilt, psi) in degrees to rotation matrices.

    Follows the cryoDRGN convention: ZYZ Euler angles, transposed, with sign
    flips to account for RELION's image coordinate convention.

    Args:
        euler: (N, 3) array of [AngleRot, AngleTilt, AnglePsi] in degrees.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    N = len(euler)
    a = euler[:, 0] * np.pi / 180.0 # AngleRot
    b = euler[:, 1] * np.pi / 180.0 # AngleTilt
    y = euler[:, 2] * np.pi / 180.0 # AnglePsi

    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cy, sy = np.cos(y), np.sin(y)

    # Ra: rotation about Z by a
    Ra = np.zeros((N, 3, 3))
    Ra[:, 0, 0] = ca
    Ra[:, 0, 1] = -sa
    Ra[:, 1, 0] = sa
    Ra[:, 1, 1] = ca
    Ra[:, 2, 2] = 1.0

    # Rb: rotation about Y by b
    Rb = np.zeros((N, 3, 3))
    Rb[:, 0, 0] = cb
    Rb[:, 0, 2] = sb
    Rb[:, 1, 1] = 1.0
    Rb[:, 2, 0] = -sb
    Rb[:, 2, 2] = cb

    # Ry: rotation about Z by y
    Ry = np.zeros((N, 3, 3))
    Ry[:, 0, 0] = cy
    Ry[:, 0, 1] = -sy
    Ry[:, 1, 0] = sy
    Ry[:, 1, 1] = cy
    Ry[:, 2, 2] = 1.0

    # R = (Ra @ Rb @ Ry)^T
    Ra_t = Ra.transpose(0, 2, 1)
    Rb_t = Rb.transpose(0, 2, 1)
    Ry_t = Ry.transpose(0, 2, 1)
    rmat = Ry_t @ Rb_t @ Ra_t

    # Sign flips for RELION convention
    rmat[:, 0, 2] *= -1
    rmat[:, 2, 0] *= -1
    rmat[:, 1, 2] *= -1
    rmat[:, 2, 1] *= -1

    return rmat


# ============================================================================
# CLI
# ============================================================================


@app.command()
def main(
    star_path: Annotated[Path, typer.Option("--star", "-s", help="Input RELION .star file", exists=True)],
    poses_path: Annotated[Path | None, typer.Option("--poses", "-p", help="Output poses pickle file")] = None,
    ctf_path: Annotated[Path | None, typer.Option("--ctf", "-c", help="Output CTF pickle file")] = None,
    D: Annotated[int | None, typer.Option("--D", "-D", help="Override image size (pixels)")] = None,
    Apix: Annotated[float | None, typer.Option("--Apix", help="Override pixel size (A/px)")] = None,
    kv: Annotated[float | None, typer.Option("--kv", help="Override voltage (kV)")] = None,
    cs: Annotated[float | None, typer.Option("--cs", help="Override spherical aberration (mm)")] = None,
    w: Annotated[float | None, typer.Option("-w", help="Override amplitude contrast")] = None,
    ps: Annotated[float | None, typer.Option("--ps", help="Override phase shift (degrees)")] = None,
) -> None:
    """Parse poses and/or CTF from a RELION .star file into pickle files."""

    if poses_path is None and ctf_path is None:
        print("Error: at least one of --poses or --ctf must be specified.")
        raise typer.Exit(code=1)

    print(f"Parsing star file: {star_path}")
    blocks = _parse_star_file(star_path)

    # Identify main data block
    main_block = blocks.get("data_particles") or blocks.get("data_")
    if main_block is None:
        print("Error: no data block found in star file.")
        raise typer.Exit(code=1)

    N = len(main_block["data"])
    is_relion31 = "data_optics" in blocks
    print(f"  RELION {'3.1' if is_relion31 else '3.0'} format, {N} particles")

    # ------------------------------------------------------------------
    # Parse poses
    # ------------------------------------------------------------------
    if poses_path is not None:
        # Euler angles
        rot_col = _get_field_values(blocks, "_rlnAngleRot", N)
        tilt_col = _get_field_values(blocks, "_rlnAngleTilt", N)
        psi_col = _get_field_values(blocks, "_rlnAnglePsi", N)

        if rot_col is None or tilt_col is None or psi_col is None:
            print("Error: Euler angle columns (_rlnAngleRot/Tilt/Psi) not found in star file.")
            raise typer.Exit(code=1)

        euler = np.stack([rot_col, tilt_col, psi_col], axis=1)
        rotations = _R_from_relion(euler).astype(np.float32)

        # Translations
        if is_relion31:
            # RELION 3.1: translations in Angstroms
            tx = _get_field_values(blocks, "_rlnOriginXAngst", N)
            ty = _get_field_values(blocks, "_rlnOriginYAngst", N)
            apix = _get_field_values(blocks, "_rlnImagePixelSize", N)
            if tx is not None and ty is not None and apix is not None:
                trans = np.stack([tx / apix, ty / apix], axis=1)
            else:
                print("Warning: translation columns not found, using zeros.")
                trans = np.zeros((N, 2))
        else:
            # RELION 3.0: translations in pixels
            tx = _get_field_values(blocks, "_rlnOriginX", N)
            ty = _get_field_values(blocks, "_rlnOriginY", N)
            if tx is not None and ty is not None:
                trans = np.stack([tx, ty], axis=1)
            else:
                print("Warning: translation columns not found, using zeros.")
                trans = np.zeros((N, 2))

        # Get D for normalization
        resolution = D
        if resolution is None:
            d_col = _get_field_values(blocks, "_rlnImageSize", N)
            if d_col is not None:
                resolution = int(d_col[0])
            else:
                print("Error: image size (D) not found in star file. Use --D to specify.")
                raise typer.Exit(code=1)

        # Normalize translations by D to get fractional coordinates
        trans = (trans / resolution).astype(np.float32)

        poses_path.parent.mkdir(parents=True, exist_ok=True)
        with open(poses_path, "wb") as f:
            pickle.dump((rotations, trans), f)
        print(f"  Poses saved to {poses_path} (rotations: {rotations.shape}, translations: {trans.shape})")

    # ------------------------------------------------------------------
    # Parse CTF
    # ------------------------------------------------------------------
    if ctf_path is not None:
        # Image size
        img_size = D
        if img_size is None:
            d_col = _get_field_values(blocks, "_rlnImageSize", N)
            if d_col is not None:
                img_size = int(d_col[0])
            else:
                print("Error: image size (D) not found in star file. Use --D to specify.")
                raise typer.Exit(code=1)

        # Pixel size
        pixel_size = Apix
        if pixel_size is None:
            apix_col = _get_field_values(blocks, "_rlnImagePixelSize", N)
            if apix_col is not None:
                pixel_size = float(apix_col[0])
            else:
                print("Error: pixel size not found in star file. Use --Apix to specify.")
                raise typer.Exit(code=1)

        # CTF fields
        df1 = _get_field_values(blocks, "_rlnDefocusU", N)
        df2 = _get_field_values(blocks, "_rlnDefocusV", N)
        dfang = _get_field_values(blocks, "_rlnDefocusAngle", N)
        volt = _get_field_values(blocks, "_rlnVoltage", N)
        sph_ab = _get_field_values(blocks, "_rlnSphericalAberration", N)
        amp = _get_field_values(blocks, "_rlnAmplitudeContrast", N)
        phaseshift = _get_field_values(blocks, "_rlnPhaseShift", N)

        if df1 is None or df2 is None:
            print("Error: defocus columns (_rlnDefocusU/V) not found in star file.")
            raise typer.Exit(code=1)

        # Build (N, 9) CTF array
        ctf = np.zeros((N, 9), dtype=np.float32)
        ctf[:, 0] = img_size
        ctf[:, 1] = pixel_size
        ctf[:, 2] = df1
        ctf[:, 3] = df2
        ctf[:, 4] = dfang if dfang is not None else 0.0
        ctf[:, 5] = kv if kv is not None else (volt if volt is not None else 300.0)
        ctf[:, 6] = cs if cs is not None else (sph_ab if sph_ab is not None else 2.7)
        ctf[:, 7] = w if w is not None else (amp if amp is not None else 0.1)
        ctf[:, 8] = ps if ps is not None else (phaseshift if phaseshift is not None else 0.0)

        ctf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ctf_path, "wb") as f:
            pickle.dump(ctf, f)
        print(f"  CTF saved to {ctf_path} (shape: {ctf.shape})")

    print("Done.")


if __name__ == "__main__":
    app()
