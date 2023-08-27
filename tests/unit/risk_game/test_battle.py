from typing import Tuple, Iterable, Callable
import pytest
from scenarios.risk_game.battle import battle


def gen_rand_func_const(x: int):
    return lambda: x


_CASES: Iterable[Tuple[int,int,Callable[[],int],Callable[[],int],int,int]] = [
    (
        3,
        2,
        gen_rand_func_const(1),
        gen_rand_func_const(1),
        2,
        0
    ),
    (
        3,
        2,
        gen_rand_func_const(2),
        gen_rand_func_const(1),
        0,
        2
    ),
]


@pytest.mark.parametrize(("atks","defs","atk_rand_func","def_rand_func","exp_a","exp_d"), _CASES)
def test_results(atks: int, defs: int, atk_rand_func: Callable[[],int], def_rand_func: Callable[[],int], exp_a: int, exp_d: int) -> None:
    out_a, out_d = battle(atks, defs, atk_rand_func, def_rand_func)
    assert out_a == exp_a
    assert out_d == exp_d
