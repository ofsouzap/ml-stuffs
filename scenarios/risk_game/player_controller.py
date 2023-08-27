from typing import Callable, Iterable, NamedTuple, Optional
from abc import ABC, abstractmethod
from .game import Game


class AttackAction(NamedTuple):
    from_territory: int
    to_territory: int
    attackers: int


class TroopRelocateAction(NamedTuple):
    from_territory: int
    to_territory: int
    troop_count: int


class PlayerControllerBase(ABC):

    def decide_initial_placing_board_occupy_territory(self, game: Game) -> int:
        """Tells the player to choose a single unoccupied territory to occupy during the initial placement phase.

Parameters:

    game - the instance of the running game

Returns:

    territory - the territory the player has chosen to place a troop in to occupy
"""
        raise NotImplementedError()

    def decide_initial_placing_troop_placement_territory(self, game: Game) -> int:
        """Tells the player to choose a single territory to place another troop in during the initial placement phase.
    
Parameters:

    game - the instance of the running game

Returns:

    territory - the territory the player has chosen to place a troop in
"""
        raise NotImplementedError()


    def decide_troop_placement_territories(self, game: Game, troop_count: int) -> Iterable[int]:
        """Tells the player to choose where to place the given number of troops.

Parameters:

    game - the instance of the running game

    troop_count - the number of troops the player must place

Returns:

    territories - an iterable of the territories the player has chosen to place their troops in. \
The length of `territories` will be `troop_count` and repeated values are allowed
"""
        raise NotImplementedError()

    def decide_attack_action(self, game: Game) -> Optional[AttackAction]:
        """Tells the player to choose where they want to attack next or if they want to stop attacking now.

Parameters:

    game - the instance of the running game

Returns:

    attack_action - the attacking action the player has chosen to perform. \
If None then this means that the player has chosen not to relocate
"""
        raise NotImplementedError()

    def decide_defender_count(self, game: Game, attack_action: AttackAction) -> int:
        """Tells the player to choose how many defenders they want to use to defend against an attack.

Parameters:

    game - the instance of the running game

    attack_action - the attack that the player must defend against

Returns:

    defenders - the number of defenders the player has chosen to use
"""
        raise NotImplementedError()

    def decide_troop_relocate(self, game: Game) -> Optional[TroopRelocateAction]:
        """Tells the player to choose where they want to relocate troops from and to or if they don't want to relocate any troops.

Parameters:

    game - the instance of the running game

Returns:

    troop_relocate_action - the relocation action the player has chosen to perform. \
If None then this means that the player has chosen not to relocate
"""
        raise NotImplementedError()


class ComputerPlayerController(PlayerControllerBase):
    pass  # TODO
