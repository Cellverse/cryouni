from __future__ import annotations

from functools import lru_cache


@lru_cache
def _get_last_n_dims(n: int) -> tuple[int, ...]:
    """
    Get the tuple of the last n dimensions.

    Args:
        `n` (int): Number of spatial dimensions.

    Returns:
        (tuple[int, ...]): Tuple of dimensions to flip.
            Empty tuple if `n` is non-positive.
    """
    return tuple(range(-n, 0))
