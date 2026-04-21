"""TSP solver using 2-opt algorithm."""

import numpy as np


def solve_tsp(points: np.ndarray) -> np.ndarray:
    """
    Args:
        `points` (np.ndarray): Points to visit of shape [N, D].

    Returns:
        `ordered_points` (np.ndarray): Reordered points of shape [N, D].
    """
    n = len(points)
    if n <= 2:
        return points

    dist_matrix = np.linalg.norm(points[:, np.newaxis] - points, axis=2)

    path = [0]
    unvisited = set(range(1, n))
    while unvisited:
        last = path[-1]
        next_node = min(unvisited, key=lambda x: dist_matrix[last][x])
        path.append(next_node)
        unvisited.remove(next_node)

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                d_old = dist_matrix[path[i - 1], path[i]] + dist_matrix[path[j], path[(j + 1) % n]]
                d_new = dist_matrix[path[i - 1], path[j]] + dist_matrix[path[i], path[(j + 1) % n]]

                if d_new < d_old:
                    path[i : j + 1] = path[i : j + 1][::-1]
                    improved = True
        if not improved:
            break

    return points[path]
