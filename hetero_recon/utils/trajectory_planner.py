"""Trajectory planning through latent space density landscape."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import skfmm
import torch

from hetero_recon.utils.kde_gpu import GaussianKDE
from hetero_recon.utils.tsp import solve_tsp


class TrajectoryPlanner:
    """
    Plan trajectories through density landscapes in reduced space.

    Supports multiple planning methods:
    - "gradient": Gradient-driven path refinement (N-D)
    - "fmm": Fast Marching Method / Eikonal solver (2D only)
    - "linear": Simple linear interpolation (N-D)

    Args:
        `points` (np.ndarray): Data points in reduced space of shape [N, D].
        `waypoints` (np.ndarray, optional): Waypoints to connect of shape [M, D].
            If None, uses auto-detected peaks.
        `peaks` (np.ndarray, optional): Pre-computed peaks of shape [K, D].
            Used when waypoints is None.
    """

    def __init__(
        self,
        points: np.ndarray,
        waypoints: np.ndarray | None = None,
        peaks: np.ndarray | None = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.points = np.asarray(points)
        self.n, self.d = self.points.shape

        # Initialize KDE for density computations
        self.kde = GaussianKDE(torch.as_tensor(self.points, device=self.device))

        # Store waypoints or use peaks
        if waypoints is not None:
            self._waypoints = np.asarray(waypoints)
        elif peaks is not None:
            self._waypoints = np.asarray(peaks)
        else:
            self._waypoints = None

        self._trajectory = None
        self._ordered_waypoints = None

    def plan(
        self,
        method: str = "gradient",
        num_steps: int = 100,
        close: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Plan trajectory through waypoints.

        Args:
            `method` (str): Planning method - "gradient", "fmm", or "linear".
            `num_steps` (int): Total trajectory points.
            `close` (bool): Create closed loop if True.
            `**kwargs`: Method-specific parameters.

        Returns:
            `trajectory` (np.ndarray): Trajectory of shape [num_steps, D].
        """
        if self._waypoints is None:
            raise ValueError("No waypoints provided. Set waypoints or peaks in constructor.")

        if len(self._waypoints) < 2:
            raise ValueError("Need at least 2 waypoints to plan trajectory.")

        # Order waypoints using TSP if requested
        self._ordered_waypoints = solve_tsp(self._waypoints)

        # Plan based on method
        if method == "gradient":
            self._trajectory = self._plan_gradient(num_steps, close, **kwargs)
        elif method == "fmm":
            if self.d > 3:
                raise ValueError(f"FMM method only supports 2D/3D, got {self.d}D.")
            self._trajectory = self._plan_fmm(num_steps, close, **kwargs)
        elif method == "linear":
            self._trajectory = self._plan_linear(num_steps, close)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'gradient', 'fmm', or 'linear'.")

        return self._trajectory

    def _plan_gradient(
        self,
        num_steps: int,
        close: bool,
        iterations: int = 100,
        alpha: float = 0.3,
        beta: float = 0.5,
    ) -> np.ndarray:
        """Plan trajectory using gradient-driven path refinement."""
        pts = self._ordered_waypoints
        num_segments = len(pts) if close else len(pts) - 1
        steps_per_segment = num_steps // num_segments

        segments = []
        for i in range(num_segments):
            end_idx = (i + 1) % len(pts) if close else i + 1
            segment = self._gradient_driven_path(
                pts[i],
                pts[end_idx],
                steps=steps_per_segment,
                iterations=iterations,
                alpha=alpha,
                beta=beta,
            )
            segments.append(segment)

        full_path = np.concatenate(segments, axis=0)
        return self._resample_path(full_path, num_steps)

    def _plan_fmm(self, num_steps: int, close: bool, resolution: int = 128) -> np.ndarray:
        """Plan trajectory using Fast Marching Method (2D/3D)."""
        pts = self._ordered_waypoints
        num_segments = len(pts) if close else len(pts) - 1
        steps_per_segment = num_steps // num_segments

        segments = []
        for i in range(num_segments):
            end_idx = (i + 1) % len(pts) if close else i + 1
            segment = self._solve_eikonal_fmm(
                pts[i],
                pts[end_idx],
                resolution=resolution,
                steps=steps_per_segment,
            )
            segments.append(segment)

        full_path = np.concatenate(segments, axis=0)
        return self._resample_path(full_path, num_steps)

    def _plan_linear(self, num_steps: int, close: bool) -> np.ndarray:
        """Plan trajectory using linear interpolation."""
        pts = self._ordered_waypoints
        num_segments = len(pts) if close else len(pts) - 1
        steps_per_segment = num_steps // num_segments

        segments = []
        for i in range(num_segments):
            end_idx = (i + 1) % len(pts) if close else i + 1
            t = np.linspace(0, 1, steps_per_segment).reshape(-1, 1)
            segment = pts[i] + t * (pts[end_idx] - pts[i])
            segments.append(segment)

        full_path = np.concatenate(segments, axis=0)
        return self._resample_path(full_path, num_steps)

    def _gradient_driven_path(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        steps: int = 30,
        iterations: int = 100,
        alpha: float = 0.3,
        beta: float = 0.5,
    ) -> np.ndarray:
        """
        Refine path using density gradients and elastic constraints.

        Args:
            `p1` (np.ndarray): Start point of shape [D].
            `p2` (np.ndarray): End point of shape [D].
            `steps` (int): Number of intermediate points.
            `iterations` (int): Number of refinement iterations.
            `alpha` (float): Gradient ascent strength.
            `beta` (float): Elastic force strength.

        Returns:
            `refined_path` (np.ndarray): Refined path of shape [steps, D].
        """
        t = torch.linspace(0, 1, steps, device=self.device).view(-1, 1)
        p1_t = torch.from_numpy(p1).to(self.device).float()
        p2_t = torch.from_numpy(p2).to(self.device).float()
        refined_path = p1_t + t * (p2_t - p1_t)

        data_t = torch.from_numpy(self.points).to(self.device).float()
        data_scale = torch.std(data_t, dim=0, unbiased=False).mean()

        for i in range(iterations):
            mid_points = refined_path[1 :-1]
            grads = self.kde.pdf_gradient(mid_points)
            grad_norms = torch.norm(grads, dim=1, keepdim=True)
            unit_grads = grads / (grad_norms + 1e-8)

            # Anneal alpha over iterations
            curr_alpha = alpha * (1.0 - i / iterations)

            elastic = (refined_path[:-2] + refined_path[2 :]) / 2.0 - mid_points
            refined_path[1 :-1] = mid_points + curr_alpha * data_scale * unit_grads + beta * elastic

        return refined_path.detach().cpu().numpy()

    def _solve_eikonal_fmm(
        self,
        start_pt: np.ndarray,
        end_pt: np.ndarray,
        resolution: int = 128,
        steps: int = 200,
    ) -> np.ndarray:
        """
        Find optimal path by solving the Eikonal equation (2D/3D).

        Cost function is derived from negative log-density (maximum likelihood path).

        Args:
            `start_pt` (np.ndarray): Start point of shape [D] (D=2 or 3).
            `end_pt` (np.ndarray): End point of shape [D] (D=2 or 3).
            `resolution` (int): Grid resolution per dimension for FMM solver.
            `steps` (int): Maximum backtracking steps.

        Returns:
            `path` (np.ndarray): Optimal path of shape [N, D].
        """
        d = self.d
        if d > 3:
            raise ValueError(f"FMM method only supports 2D/3D, got {d}D.")

        # Prepare grid and cost function
        # grid shape: [R, R, 2] for 2D, [R, R, R, 3] for 3D
        grid = self.kde.get_grid(resolution=resolution)
        grid_np = grid.cpu().numpy()

        # Extract 1D axis coordinates for each dimension
        axis_coords = []
        for dim in range(d):
            idx = tuple(0 if i != dim else slice(None) for i in range(d)) + (dim,)
            axis_coords.append(grid_np[idx])

        log_Z = self.kde.logpdf(grid).cpu().numpy()

        # Cost: High density = Low cost
        cost = -log_Z
        cost = cost - cost.min() + 0.1

        # Setup FMM: place source at grid cell closest to start_pt
        phi = np.ones_like(cost)
        start_idx = tuple(int(np.abs(axis_coords[dim] - start_pt[dim]).argmin()) for dim in range(d))
        phi[start_idx] = 0

        dx = [float(axis_coords[dim][1] - axis_coords[dim][0]) for dim in range(d)]
        travel_time = skfmm.travel_time(phi, speed=1.0 / cost, dx=dx)

        # Compute gradients and build interpolators for each dimension
        grads_list = np.gradient(travel_time, *dx)
        interp_grads = [RegularGridInterpolator(tuple(axis_coords), g, bounds_error=False, fill_value=None) for g in grads_list]

        # Backtrack from end to start
        path = [end_pt.copy()]
        curr_p = end_pt.copy()
        dist_init = np.linalg.norm(end_pt - start_pt)
        dt = dist_init / steps

        for _ in range(steps * 5):
            p_query = curr_p.reshape(1, d)
            try:
                grad = np.array([interp(p_query)[0] for interp in interp_grads])
            except (ValueError, IndexError):
                break

            norm = np.linalg.norm(grad)
            if norm < 1e-12:
                break

            curr_p -= (grad / norm) * dt
            path.append(curr_p.copy())

            if np.linalg.norm(curr_p - start_pt) < dt * 1.5:
                path.append(start_pt)
                break

        return np.array(path[::-1])

    def _resample_path(self, path: np.ndarray, num_steps: int) -> np.ndarray:
        """Resample path to uniform distance between steps."""
        distances = np.sqrt(np.sum(np.diff(path, axis=0) ** 2, axis=1))
        cumulative_dist = np.insert(np.cumsum(distances), 0, 0)
        uniform_dist = np.linspace(0, cumulative_dist[-1], num_steps)
        return np.array([np.interp(uniform_dist, cumulative_dist, path[:, d]) for d in range(path.shape[1])]).T

    def get_trajectory(self) -> np.ndarray | None:
        """Return planned trajectory or None if not planned."""
        return self._trajectory

    def get_waypoints(self) -> np.ndarray | None:
        """Return waypoints or None."""
        return self._waypoints

    def get_ordered_waypoints(self) -> np.ndarray | None:
        """Return TSP-ordered waypoints or None."""
        return self._ordered_waypoints

    def get_path_length(self) -> float | None:
        """
        Return total path length or None if not planned.

        Returns:
            `length` (float): Total Euclidean distance along trajectory.
        """
        if self._trajectory is None:
            return None
        distances = np.sqrt(np.sum(np.diff(self._trajectory, axis=0) ** 2, axis=1))
        return np.sum(distances)

    def get_visualization_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Get data for visualization (2D/3D only).

        Returns:
            Tuple of (`grid`, `density`, `trajectory`) or None if >3D.
        """
        if self.d > 3 or self._trajectory is None:
            return None

        grid = self.kde.get_grid()
        density = self.kde.pdf(grid)

        if torch.is_tensor(grid):
            grid = grid.cpu().numpy()
        if torch.is_tensor(density):
            density = density.cpu().numpy()

        return grid, density, self._trajectory
