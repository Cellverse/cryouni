"""
Parse poses and CTF parameters from cryoSPARC .cs files.

Converts cryoSPARC .cs files into the pickle format expected by CryoUNI,
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
    help="Parse poses and CTF from cryoSPARC .cs files into CryoUNI pickle format.",
    add_completion=False,
)

# ============================================================================
# Rotation conversion (axis-angle -> rotation matrix, Rodrigues' formula)
# ============================================================================


def _axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle vectors to rotation matrices using Rodrigues' formula.

    Args:
        axis_angle: (N, 3) array where direction = axis, norm = angle.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    N = len(axis_angle)
    theta = np.linalg.norm(axis_angle, axis=1, keepdims=True) # (N, 1)

    # Avoid division by zero for near-identity rotations
    safe_theta = np.where(theta > 1e-10, theta, np.ones_like(theta))

    axis = axis_angle / safe_theta # (N, 3)

    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    cos_t = np.cos(theta[:, 0])
    sin_t = np.sin(theta[:, 0])
    one_m_cos = 1.0 - cos_t

    # Rodrigues' formula
    R = np.zeros((N, 3, 3), dtype=np.float64)
    R[:, 0, 0] = cos_t + one_m_cos * x * x
    R[:, 0, 1] = one_m_cos * x * y - sin_t * z
    R[:, 0, 2] = one_m_cos * x * z + sin_t * y
    R[:, 1, 0] = one_m_cos * y * x + sin_t * z
    R[:, 1, 1] = cos_t + one_m_cos * y * y
    R[:, 1, 2] = one_m_cos * y * z - sin_t * x
    R[:, 2, 0] = one_m_cos * z * x - sin_t * y
    R[:, 2, 1] = one_m_cos * z * y + sin_t * x
    R[:, 2, 2] = cos_t + one_m_cos * z * z

    # For near-zero rotations, use identity
    small = (theta[:, 0] < 1e-10)
    R[small] = np.eye(3)

    return R


@app.command()
def main(
    cs_path: Annotated[Path, typer.Option("--cs", "-s", help="Input cryoSPARC .cs file", exists=True)],
    poses_path: Annotated[Path | None, typer.Option("--poses", "-p", help="Output poses pickle file")] = None,
    ctf_path: Annotated[Path | None, typer.Option("--ctf", "-c", help="Output CTF pickle file")] = None,
    D: Annotated[int | None, typer.Option("--D", "-D", help="Image size in pixels (required for poses)")] = None,
    Apix: Annotated[float | None, typer.Option("--Apix", help="Override pixel size (A/px)")] = None,
    abinit: Annotated[bool, typer.Option("--abinit", help="Use ab-initio alignment fields")] = False,
    hetrefine: Annotated[bool, typer.Option("--hetrefine", help="Use heterogeneous refinement fields (shifts scaled by 2x)")] = False,
) -> None:
    """Parse poses and/or CTF from a cryoSPARC .cs file into pickle files."""

    if poses_path is None and ctf_path is None:
        print("Error: at least one of --poses or --ctf must be specified.")
        raise typer.Exit(code=1)

    print(f"Parsing cryoSPARC file: {cs_path}")
    metadata = np.load(cs_path)
    N = len(metadata)
    print(f"  {N} particles")

    # ------------------------------------------------------------------
    # Parse poses
    # ------------------------------------------------------------------
    if poses_path is not None:
        if D is None:
            # Try to get D from metadata (blob/shape for job .cs, fallback for passthrough .cs)
            for field in ["blob/shape", "location/micrograph_shape"]:
                try:
                    D = int(metadata[field][0][0])
                    print(f"  Image size D={D} (from {field})")
                    break
                except (KeyError, IndexError, ValueError):
                    continue
            if D is None:
                print("Error: image size (D) not found in .cs file. Use --D to specify.")
                raise typer.Exit(code=1)

        # Select alignment fields based on job type
        if abinit:
            pose_key = "alignments_class_0/pose"
            shift_key = "alignments_class_0/shift"
        else:
            pose_key = "alignments3D/pose"
            shift_key = "alignments3D/shift"

        # Rotations: axis-angle (N, 3) -> rotation matrix (N, 3, 3)
        try:
            aa = metadata[pose_key].astype(np.float64)
        except (KeyError, ValueError):
            print(f"Error: pose field '{pose_key}' not found. Try --abinit flag for ab-initio jobs.")
            raise typer.Exit(code=1)

        rotations = _axis_angle_to_rotation_matrix(aa)
        # Transpose to match cryoDRGN convention
        rotations = rotations.transpose(0, 2, 1).astype(np.float32)

        # Translations
        try:
            trans = metadata[shift_key].astype(np.float32)
        except (KeyError, ValueError):
            print("Warning: shift field not found, using zeros.")
            trans = np.zeros((N, 2), dtype=np.float32)

        if hetrefine:
            trans *= 2.0

        # Normalize by D to get fractional coordinates
        trans = (trans / D).astype(np.float32)

        poses_path.parent.mkdir(parents=True, exist_ok=True)
        with open(poses_path, "wb") as f:
            pickle.dump((rotations, trans), f)
        print(f"  Poses saved to {poses_path} (rotations: {rotations.shape}, translations: {trans.shape})")

    # ------------------------------------------------------------------
    # Parse CTF
    # ------------------------------------------------------------------
    if ctf_path is not None:
        # Image size
        if D is not None:
            img_size = D
        else:
            img_size = None
            for field in ["blob/shape", "location/micrograph_shape"]:
                try:
                    img_size = int(metadata[field][0][0])
                    break
                except (KeyError, IndexError, ValueError):
                    continue
            if img_size is None:
                print("Error: image size (D) not found. Use --D to specify.")
                raise typer.Exit(code=1)

        # Pixel size
        if Apix is not None:
            pixel_size = Apix
        else:
            pixel_size = None
            for field in ["blob/psize_A", "alignments3D/psize_A", "location/micrograph_psize_A"]:
                try:
                    pixel_size = float(metadata[field][0])
                    print(f"  Pixel size {pixel_size} A/px (from {field})")
                    break
                except (KeyError, IndexError, ValueError):
                    continue
            if pixel_size is None:
                print("Error: pixel size not found in .cs file. Use --Apix to specify.")
                raise typer.Exit(code=1)

        # CTF fields
        try:
            df1 = metadata["ctf/df1_A"].astype(np.float64)
            df2 = metadata["ctf/df2_A"].astype(np.float64)
            dfang_rad = metadata["ctf/df_angle_rad"].astype(np.float64)
            volt = metadata["ctf/accel_kv"].astype(np.float64)
            sph_ab = metadata["ctf/cs_mm"].astype(np.float64)
            amp = metadata["ctf/amp_contrast"].astype(np.float64)
            phase_rad = metadata["ctf/phase_shift_rad"].astype(np.float64)
        except KeyError as e:
            print(f"Error: CTF field {e} not found in .cs file.")
            raise typer.Exit(code=1)

        # Build (N, 9) CTF array — angles converted from radians to degrees
        ctf = np.zeros((N, 9), dtype=np.float32)
        ctf[:, 0] = img_size
        ctf[:, 1] = pixel_size
        ctf[:, 2] = df1
        ctf[:, 3] = df2
        ctf[:, 4] = dfang_rad * 180.0 / np.pi
        ctf[:, 5] = volt
        ctf[:, 6] = sph_ab
        ctf[:, 7] = amp
        ctf[:, 8] = phase_rad * 180.0 / np.pi

        ctf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ctf_path, "wb") as f:
            pickle.dump(ctf, f)
        print(f"  CTF saved to {ctf_path} (shape: {ctf.shape})")

    print("Done.")


if __name__ == "__main__":
    app()
