from typing import Tuple, List, Callable, Optional
from random import randint


def roll_dice() -> int:
    return randint(1,6)


def battle(attackers: int,
           defenders: int,
           atk_rand_func: Optional[Callable[[], int]] = None,
           def_rand_func: Optional[Callable[[], int]] = None) -> Tuple[int,int]:
    """Simulates a single battle and returns the result

Parameters:

    attackers - the number of attacking troops

    defenders - the number of defending troops

    atk_rand_func (optional) - the function to call for simulating the dice rolling for the attackers. Defaults to an unbiased 6-sided dice function

    def_rand_func (optional) - the function to call for simulating the dice rolling for the defenders. Defaults to an unbiased 6-sided dice function

Returns:

    attackers_dead - how many attackers die

    defenders_dead - how many defenders die
"""

    if atk_rand_func is None:
        atk_rand_func = roll_dice

    if def_rand_func is None:
        def_rand_func = roll_dice

    values_taken = min(attackers, defenders)

    atk_rolls = sorted([atk_rand_func() for _ in range(attackers)], reverse=True)[:values_taken]
    def_rolls = sorted([def_rand_func() for _ in range(defenders)], reverse=True)[:values_taken]

    roll_results: List[bool] = [atk > def_ for atk, def_ in zip(atk_rolls, def_rolls)]
    """Each is True if the attacker won and False if the defender won"""

    attackers_dead = len(list(filter(lambda x: not x, roll_results)))
    defenders_dead = len(list(filter(lambda x: x, roll_results)))

    return attackers_dead, defenders_dead
