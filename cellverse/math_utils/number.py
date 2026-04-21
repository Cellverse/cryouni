"""
This module provides functions for working with non-negative integers.

Note:
    Undefined behavior for negative numbers.
"""

__all__ = [
    "get_next_even",
    "get_next_odd",
    "get_prev_even",
    "get_prev_odd",
    "is_even",
    "is_odd",
    "get_nyquist_index",
]


def is_even(n: int) -> bool:
    return n % 2 == 0


def is_odd(n: int) -> bool:
    return n % 2 == 1


def get_next_even(n: int) -> int:
    """
    Example:
        >>> get_next_even(1)
        >>> 2
        >>> get_next_even(2)
        >>> 2
    """
    return n if is_even(n) else n + 1


def get_next_odd(n: int) -> int:
    """
    Example:
        >>> get_next_odd(1)
        >>> 1
        >>> get_next_odd(2)
        >>> 3
    """
    return n if is_odd(n) else n + 1


def get_prev_even(n: int) -> int:
    """
    Example:
        >>> get_prev_even(1)
        >>> 0
        >>> get_prev_even(2)
        >>> 2
    """
    return n if is_even(n) else n - 1


def get_prev_odd(n: int) -> int:
    """
    Example:
        >>> get_prev_odd(1)
        >>> 1
        >>> get_prev_odd(2)
        >>> 1
    """
    return n if is_odd(n) else n - 1


def get_ceil_div(n: int, d: int) -> int:
    """
    Example:
        >>> get_ceil_div(15, 4)
        >>> 4
        >>> get_ceil_div(16, 4)
        >>> 4
        >>> get_ceil_div(17, 4)
        >>> 5
    """
    return (n + d - 1) // d


def get_next_power_of_2(n: int) -> int:
    """
    Undefined behavior for non-positive numbers.

    Example:
        >>> get_next_power_of_2(1)
        >>> 1
        >>> get_next_power_of_2(2)
        >>> 2
        >>> get_next_power_of_2(3)
        >>> 4
    """
    return 1 << (n - 1).bit_length()


def get_nyquist_index(shape: tuple[int, ...]) -> int:
    """
    Gets the Nyquist frequency index for a given shape,
    which is defined as the half of the minimum dimension size.
    """
    return min(shape) // 2
