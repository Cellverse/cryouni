"""
This module provides functions to read and write cryo-EM data files.

Currently, the following file formats are supported:
    - particle stack in traditional MRC/MRCs format
    - particle stack in CryoCRAB HDF5 format
    - poses / ctfs / indices in CryoDRGN pickle format
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
from typing import Any

import h5py
import mrcfile
import numpy as np
import torch

from cellverse.cryoem.ctf import (
    CTFParameters,
    ImageParameters,
)
from cellverse.math_utils.number import get_next_odd

from . import mrc, starfile

__all__ = []


def read_cryodrgn_particle(particle_path: os.PathLike, particle_dir: os.PathLike, lazy: bool = False) -> np.ndarray | list[mrc.LazyImage]:

    if particle_path.suffix == '.txt':
        particle_images = mrc.parse_mrc_list(particle_path, lazy=lazy)
    elif particle_path.suffix == '.star':
        datadir = os.path.dirname(particle_path) if particle_dir is None else particle_dir
        particle_images = starfile.Starfile.load(particle_path, relion31=True).get_particles(datadir=datadir, lazy=lazy)
    elif particle_path.suffix == '.cs':
        particle_images = starfile.csparc_get_particles(particle_path, datadir=particle_dir, lazy=lazy)
    else:
        particle_images, _ = mrc.parse_mrc(particle_path, lazy=lazy)

    return particle_images


class LazyH5Dataset:
    """
    A stateless proxy class for HDF5 datasets, designed for multi-process DataLoaders.

    Unlike standard h5py usage, this class DOES NOT maintain an open file handle.
    Instead, it opens and closes the file for every single read operation.

    This completely eliminates file handle conflicts (SIGABRT/SIGBUS) and race conditions
    when using PyTorch's DataLoader with num_workers > 0.
    """

    def __init__(self, path: str | Path, key: str, shape: tuple = None):
        """
        Args:
            path: Path to the .h5 file.
            key: The dataset key (e.g., 'particles').
            shape: Pre-fetched shape of the dataset. Providing this is highly recommended
                   to avoid opening the file just to check len().
        """
        self.path = str(path)
        self.key = key
        # Cache the shape to optimize __len__ calls
        self._shape = shape

    def __getitem__(self, idx):
        """
        Reads data at the specified index.
        CRITICAL: The file is opened and closed immediately within this method.
        """
        # Open the file in read-only mode using a context manager.
        # This ensures the file handle is unique to this process and this specific call.
        with h5py.File(self.path, 'r') as f:
            # Read the specific slice/index.
            # accessing f[self.key][idx] loads the data into memory (numpy array).
            data = f[self.key][idx]

        # When the 'with' block exits, the file is automatically closed.
        # The 'data' variable remains valid in memory.
        return data

    def __len__(self):
        # Use cached shape if available to save I/O overhead
        if self._shape is not None:
            return self._shape[0]

        # Fallback: temporarily open the file to check length
        with h5py.File(self.path, 'r') as f:
            return f[self.key].shape[0]

    @property
    def shape(self):
        # Return cached shape if available
        if self._shape is not None:
            return self._shape

        # Fallback: check shape from file
        with h5py.File(self.path, 'r') as f:
            return f[self.key].shape

    # Default pickling behavior is safe here because we do not store
    # any open file handles (pointers), only strings and tuples.
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)


def read_h5_particle(particle_path: os.PathLike,
                     lazy: bool = False) -> tuple[Any, torch.Tensor, torch.Tensor, ImageParameters, CTFParameters]:
    """
    Modified version: Supports Lazy mode, avoiding multi-process handle conflicts.
    """
    particle_path = Path(particle_path)
    assert particle_path.suffix == ".h5", "File must be in HDF5 format."

    # Use with statement to ensure file is closed immediately after reading metadata
    with h5py.File(particle_path, "r") as f:
        # 1. Read metadata (CTF, Poses, Params)
        # Note: Must use [()] or [:] to copy data to memory, otherwise data is lost after file close

        # read image_params
        image_size = f["ctf/imgsize_pix"][0]
        pixel_size_angstrom = f["ctf/psize_A"][0]
        image_params = ImageParameters(
            image_size=get_next_odd(image_size),
            pixel_size_angstrom=pixel_size_angstrom,
        )

        # read ctf_params
        defocus_1_angstrom = torch.from_numpy(f["ctf/df1_A"][()]).float()
        defocus_2_angstrom = torch.from_numpy(f["ctf/df2_A"][()]).float()
        defocus_angle_degrees = torch.from_numpy(f["ctf/dfang_deg"][()]).float()
        voltage_kilovolts = torch.from_numpy(f["ctf/vol_kv"][()]).float()
        spherical_aberration_millimeters = torch.from_numpy(f["ctf/cs_mm"][()]).float()
        amplitude_contrast = torch.from_numpy(f["ctf/w"][()]).float()
        phase_shift_degrees = torch.from_numpy(f["ctf/phase_shift_deg"][()]).float()

        ctf_params = CTFParameters(
            defocus_1_angstrom=defocus_1_angstrom,
            defocus_2_angstrom=defocus_2_angstrom,
            defocus_angle_degrees=defocus_angle_degrees,
            voltage_kilovolts=voltage_kilovolts,
            spherical_aberration_millimeters=spherical_aberration_millimeters,
            amplitude_contrast=amplitude_contrast,
            phase_shift_degrees=phase_shift_degrees,
        )

        # read poses
        rotations = torch.from_numpy(f["poses/R"][()]).float()
        translations = torch.from_numpy(f["poses/T"][()]).float()

        # 2. Process image data
        if lazy:
            particles_shape = f["particles"].shape
            # Lazy mode: Return None, do not return open Dataset object!
            # We will reopen it later in __getitem__
            particle_images = LazyH5Dataset(particle_path, "particles", shape=particles_shape)
        else:
            # Eager mode: Read all data directly into memory (RAM)
            # Your server memory is large enough (e.g. 500G), this is usually faster
            particle_images = f["particles"][:]

    # File is closed now, safe for Worker processes
    return particle_images, rotations, translations, image_params, ctf_params


def read_pkl_ctf(ctf_path: os.PathLike) -> tuple[ImageParameters, CTFParameters]:
    """
    Read CTF parameters from a CryoDRGN pickle file.

    Args:
        ctf_path (os.PathLike): Path to the pickle file.

    Returns:
        tuple[ImageParameters, CTFParameters]: A tuple containing:
            - image_params (ImageParameters): The image parameters including size and pixel size.
            - ctf_params (CTFParameters): The CTF parameters including defocus and other CTF-related info.
    """

    ctf_path = Path(ctf_path)
    assert ctf_path.suffix == ".pkl", "File must be in pickle format."

    with open(ctf_path, "rb") as f:
        ctf = pickle.load(f)

        # read image_params
        image_size = int(ctf[0, 0])
        pixel_size_angstrom = ctf[0, 1]
        image_params = ImageParameters(
            image_size=get_next_odd(image_size),
            pixel_size_angstrom=pixel_size_angstrom,
        )

        # read ctf_params
        defocus_1_angstrom = torch.from_numpy(ctf[:, 2]).float()
        defocus_2_angstrom = torch.from_numpy(ctf[:, 3]).float()
        defocus_angle_degrees = torch.from_numpy(ctf[:, 4]).float()
        voltage_kilovolts = torch.from_numpy(ctf[:, 5]).float()
        spherical_aberration_millimeters = torch.from_numpy(ctf[:, 6]).float()
        amplitude_contrast = torch.from_numpy(ctf[:, 7]).float()
        phase_shift_degrees = torch.from_numpy(ctf[:, 8]).float()

        ctf_params = CTFParameters(
            defocus_1_angstrom=defocus_1_angstrom,
            defocus_2_angstrom=defocus_2_angstrom,
            defocus_angle_degrees=defocus_angle_degrees,
            voltage_kilovolts=voltage_kilovolts,
            spherical_aberration_millimeters=spherical_aberration_millimeters,
            amplitude_contrast=amplitude_contrast,
            phase_shift_degrees=phase_shift_degrees,
        )

    return image_params, ctf_params


def read_pkl_poses(poses_path: os.PathLike) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read poses from a CryoDRGN pickle file.

    Args:
        poses_path (os.PathLike): Path to the pickle file.

    Note: The returned translations are normalized by the image size, which are in range [-1.0, 1.0].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - rotations (torch.Tensor): The rotation matrices of shape (N, 3, 3).
            - translations (torch.Tensor): The translation vectors of shape (N, 2,).
    """

    poses_path = Path(poses_path)
    assert poses_path.suffix == ".pkl", "File must be in pickle format."

    with open(poses_path, "rb") as f:
        rotations, translations = pickle.load(f)

    rotations = torch.from_numpy(rotations).float()       # (N, 3, 3)
    translations = torch.from_numpy(translations).float() # (N, 2)

    assert translations.abs().max() <= 1.0, "Translations should be normalized by the image size, which are in range [-1.0, 1.0]."

    return rotations, translations


def read_pkl_indices(indices_path: os.PathLike) -> torch.Tensor:
    """
    Read particle indices from a CryoDRGN pickle file.

    Args:
        indices_path (os.PathLike): Path to the pickle file.

    Note:
        - The indices must be in CryoDRGN pickle format.

    Returns:
        torch.Tensor: The particle indices of shape (M,), where M is the number of selected
            particles.
    """

    indices_path = Path(indices_path)
    assert indices_path.suffix == ".pkl", "File must be in pickle format."

    with open(indices_path, "rb") as f:
        indices = pickle.load(f)

    indices = torch.from_numpy(indices).long()

    return indices


def read_particle_num(particle_path: os.PathLike) -> int:
    """
    Read the number of particles from a particle stack file.

    Args:
        particle_path (os.PathLike): Path to the particle stack file in MRC/MRCs or HDF5 format.

    Returns:
        int: The number of particles in the stack.
    """

    particle_path = Path(particle_path)
    assert particle_path.suffix in {".mrc", ".mrcs", ".h5"}, "File must be in MRC, MRCs, or HDF5 format."

    if particle_path.suffix in {".mrc", ".mrcs"}:
        with mrcfile.open(particle_path, mode="r", permissive=True, header_only=True) as mrc_file:
            num_particles = mrc_file.header.nz

    elif particle_path.suffix == ".h5":
        with h5py.File(particle_path, "r") as h5_datasets:
            num_particles = h5_datasets["particles"].shape[0]

    return num_particles


def save_pickle(data: Any, path: Path) -> None:
    """
    Save data to pickle file.

    Args:
        `data` (Any): Data to save.
        `path` (Path): Path to save pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: Path) -> Any:
    """
    Load data from pickle file.

    Args:
        `path` (Path): Path to pickle file.

    Returns:
        Loaded data.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_volume(path: Path, volume: np.ndarray | torch.Tensor, pixel_size_in_A: float) -> None:
    """
    Write a 3D numpy array to an MRC file with specified voxel size.

    Args:
        `path` (Path): The path to the MRC file to be written.
        `volume` (np.ndarray | torch.Tensor): The 3D numpy array to be written.
        `pixel_size_in_A` (float): The voxel size in Angstroms.
    """
    if torch.is_tensor(volume):
        volume = volume.detach().cpu().numpy()

    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(volume)
        mrc._set_voxel_size(pixel_size_in_A, pixel_size_in_A, pixel_size_in_A)
