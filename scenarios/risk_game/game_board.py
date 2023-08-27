from typing import Dict, Tuple, Set
from .world import World
from .setup_board import SetupBoard

class GameBoard:
    """The board when all territories are occupied"""

    def __init__(self,
                 setup_board: SetupBoard):

        if not setup_board.is_all_occupied():
            raise ValueError("Setup board is not fully occupied")

        self._world: World = setup_board.world

        self._occupations: Dict[int,Tuple[int,int]] = {}
        """territory_idx --to-> (occupying player's index, number of troops there)"""

        self._player_occupations: Dict[int, Set[int]] = {}
        """player_idx --to-> territory indexes that they occupy"""

        for territory in self._world.iterate_territories():

            occupant = setup_board.get_occupant(territory)
            assert occupant is not None

            self._occupations[territory] = (occupant, 1)

            if occupant not in self._player_occupations:
                self._player_occupations[occupant] = set()

            self._player_occupations[occupant].add(territory)

    def get_occupation(self, territory: int) -> Tuple[int,int]:
        return self._occupations[territory]

    def __getitem__(self, key) -> Tuple[int,int]:
        match key:
            case int() as territory:
                return self.get_occupation(territory)
            case _:
                raise ValueError("Invalid index")

    def get_occupier(self, territory: int) -> int:
        return self.get_occupation(territory)[0]

    def get_troop_count(self, territory: int) -> int:
        return self.get_occupation(territory)[1]

    def set_occupation(self, territory: int, occupier: int, troop_count: int) -> None:

        old_occupier_idx = self.get_occupier(territory)

        # Update self._occupations

        self._occupations[territory] = (occupier, troop_count)

        # Update self._player_occupations if needed

        if old_occupier_idx != occupier:
            self._player_occupations[old_occupier_idx].remove(territory)
            self._player_occupations[occupier].add(territory)

    def __setitem__(self, key, value) -> None:

        match key, value:

            case [int() as territory, [int() as occupier_idx, int() as troop_count]]:
                self.set_occupation(territory, occupier_idx, troop_count)

            case [int(), _]:
                raise ValueError("Invalid value")

            case [_, [int(), int()]]:
                raise ValueError("Invalid key")

            case _:
                raise ValueError("Invalid key and value")

    def add_troops(self, territory: int, amount: int) -> None:
        curr_occupation = self.get_occupation(territory)
        self.set_occupation(territory, curr_occupation[0], curr_occupation[1]+amount)

    def remove_troops(self, territory: int, amount: int) -> None:
        return self.add_troops(territory, -amount)

    def get_player_territories(self, player: int) -> Set[int]:
        return self._player_occupations[player]

    def get_player_full_continents(self, player: int) -> Set[int]:
        return set(filter(
            lambda continent: self.player_occupies_entire_continent(player, continent),
            self._world.iterate_continents()
        ))

    def player_occupies_entire_continent(self, player: int, continent: int) -> bool:
        return all(map(
            lambda territory: self.get_occupier(territory) == player,
            self._world.continent_territories[continent]
        ))

    def territories_have_continuous_route_with_same_occupier(self, territory_a: int, territory_b: int) -> bool:

        if self.get_occupier(territory_a) != self.get_occupier(territory_b):
            raise ValueError("Start and end territories don't have same occupier")

        occupier = self.get_occupier(territory_a)

        return self._world.graph.bfs_find_path(
            start=territory_a,
            stop_condition=lambda n: n == territory_b,
            valid_condition=lambda n: self.get_occupier(n) == occupier
        ) is not None
