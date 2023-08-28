from typing import List, Tuple, Dict, Iterable, Set
from graphs import UndirectedGraphBase, ArrayUndirectedGraph
from bij_map import BijMap


class World:
    """Immutable class representing a world that a game is played in"""

    def __init__(self,
                 continent_territories: Dict[str, List[str]],
                 continent_troop_gains: Dict[str, int],
                 territory_connections: Dict[str, List[str]]):

        self._continent_names = BijMap[int,str]()
        self._territory_names = BijMap[int,str]()
        self._territory_continents: Dict[int,int] = {}
        """territory --to-> continent"""
        self._continent_territories: Dict[int, Set[int]] = {}
        """continent --to-> territories"""

        # Read continents and territories

        for continent_name in continent_territories:

            # Add continent

            if self._continent_names.from_contains(continent_name):
                raise ValueError(f"Duplicate continent name found: {continent_name}")

            continent = self._continent_names.size
            self._continent_names.set_to(continent, continent_name)

            # Add territories

            for territory_name in continent_territories[continent_name]:

                if self._territory_names.from_contains(territory_name):
                    raise ValueError(f"Duplicate territory name found: {territory_name}")

                territory = self._territory_names.size
                self._territory_names.set_to(territory, territory_name)

                self._territory_continents[territory] = continent

                if continent not in self._continent_territories:
                    self._continent_territories[continent] = set()
                self._continent_territories[continent].add(territory)

        # Read territory connections

        graph_edges: List[Tuple[int,int]] = []

        for a in territory_connections:
            for b in territory_connections[a]:

                if not self._territory_names.from_contains(a):
                    raise ValueError(f"Undefined territory {a}")

                if not self._territory_names.from_contains(b):
                    raise ValueError(f"Undefined territory {b}")

                a_idx = self._territory_names.get_from(a)
                b_idx = self._territory_names.get_from(b)

                graph_edges.append((a_idx,b_idx))

        # Create graph

        self._graph: UndirectedGraphBase = ArrayUndirectedGraph(len(self._territory_names), graph_edges)

        # Read continent troop gains

        self.continent_troop_gains: Dict[int, int] = {}
        """continent_idx --to-> troops gained for owning entire continent"""

        for continent in self.iterate_continents():

            continent_name = self._continent_names.get_to(continent)

            if continent_name not in continent_troop_gains:
                raise ValueError(f"Continent {continent} missing from continent_troop_gains")
            else:
                self.continent_troop_gains[continent] = continent_troop_gains[continent_name]

    @property
    def graph(self) -> UndirectedGraphBase:
        return self._graph

    def continent_by_name(self, continent_name: str) -> int:
        if self._continent_names.from_contains(continent_name):
            return self._continent_names.get_from(continent_name)
        else:
            raise ValueError(f"Unknown continent name: {continent_name}")

    def get_continent_name(self, continent: int) -> str:
        if self._continent_names.to_contains(continent):
            return self._continent_names.get_to(continent)
        else:
            raise ValueError(f"Unknown continent: {continent}")

    def territory_by_name(self, territory_name: str) -> int:
        if self._territory_names.from_contains(territory_name):
            return self._territory_names.get_from(territory_name)
        else:
            raise ValueError(f"Unknown territory name: {territory_name}")

    def get_territory_name(self, territory: int) -> str:
        if self._territory_names.to_contains(territory):
            return self._territory_names.get_to(territory)
        else:
            raise ValueError(f"Unknown territory: {territory}")

    def get_territory_continent(self, territory: int) -> int:
        if territory in self._territory_continents:
            return self._territory_continents[territory]
        else:
            raise ValueError(f"Unknown territory: {territory}")

    def get_continent_territories(self, continent: int) -> Iterable[int]:
        if continent in self._continent_territories:
            return self._continent_territories[continent]
        else:
            raise ValueError(f"Unknown continent: {continent}")

    def is_valid_continent(self, continent: int) -> bool:
        return self._continent_names.to_contains(continent)

    def is_valid_territory(self, territory: int) -> bool:
        return self._territory_names.to_contains(territory)

    def iterate_continents(self) -> Iterable[int]:
        return self._continent_names.iterate_to()

    def iterate_territories(self) -> Iterable[int]:
        return self._territory_names.iterate_to()

    def iterate_territory_neighbours(self, territory: int) -> Iterable[int]:
        return self.graph.iterate_node_neighbours(territory)

    def territories_are_neighbours(self, territory_a: int, territory_b: int) -> bool:
        return self.graph[territory_a, territory_b]
