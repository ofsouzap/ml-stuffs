from typing import Dict, Optional
from .world import World


class SetupBoard:
    """The board when some territories are still empty"""

    def __init__(self,
                 world: World):

        self._world = world

        self._occupants: Dict[int,Optional[int]] = {}
        """territory_idx --to-> occupying player's index"""
        for territory in self._world.iterate_territories():
            self._occupants[territory] = None

    @property
    def world(self) -> World:
        return self._world

    def set_occupant(self, territory: int, player: int) -> None:

        if territory not in self._occupants:
            raise ValueError(f"Invalid territory index: {territory}")
        elif self._occupants[territory] is not None:
            raise ValueError("Territory already occupied")

        self._occupants[territory] = player

    def get_occupant(self, territory: int) -> Optional[int]:

        if territory not in self._occupants:
            raise ValueError(f"Invalid territory index: {territory}")
        else:
            return self._occupants[territory]

    def is_all_occupied(self) -> bool:
        return all(map(lambda x: x is not None, self._occupants.values()))
