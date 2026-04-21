from itertools import repeat
from typing import Callable, Iterable


def _ntuple(n: int) -> Callable[[int | tuple[int, ...]], tuple[int, ...]]:

    def parse(x: int | tuple[int, ...]) -> tuple[int, ...]:
        if isinstance(x, Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
