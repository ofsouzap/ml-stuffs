from typing import List, Tuple, Dict, DefaultDict, Iterable, Set
from collections import defaultdict
from graphs import UndirectedGraph
from bij_map import BijMap


class World:

    def __init__(self,
                 continent_territories: Dict[str, List[str]],
                 continent_troop_gains: Dict[str, int],
                 territory_connections: Dict[str, List[str]]):

        self.continent_names = BijMap[int,str]()
        self.territory_names = BijMap[int,str]()
        self.territory_continents: Dict[int,int] = {}
        """territory_idx --to-> continent_idx"""
        self.continent_territories: DefaultDict[int, Set[int]] = defaultdict(lambda: set())
        """continent_idx --to-> territory_idx"""

        # Read continents and territories

        for continent_name in continent_territories:

            # Add continent

            if self.continent_names.from_contains(continent_name):
                raise ValueError(f"Duplicate continent name found: {continent_name}")

            continent = self.continent_names.size
            self.continent_names.set_to(continent, continent_name)

            # Add territories

            for territory_name in continent_territories[continent_name]:

                if self.territory_names.from_contains(territory_name):
                    raise ValueError(f"Duplicate territory name found: {territory_name}")

                territory = self.territory_names.size
                self.territory_names.set_to(territory, territory_name)

                self.territory_continents[territory] = continent
                self.continent_territories[continent].add(territory)

        # Read territory connections

        graph_edges: List[Tuple[int,int]] = []

        for a in territory_connections:
            for b in territory_connections[a]:

                if not self.territory_names.from_contains(a):
                    raise ValueError(f"Undefined territory {a}")

                if not self.territory_names.from_contains(b):
                    raise ValueError(f"Undefined territory {b}")

                a_idx = self.territory_names.get_from(a)
                b_idx = self.territory_names.get_from(b)

                graph_edges.append((a_idx,b_idx))

        # Create graph

        self.graph = UndirectedGraph(len(self.territory_names), graph_edges)

        # Read continent troop gains

        self.continent_troop_gains: Dict[int, int] = {}
        """continent_idx --to-> troops gained for owning entire continent"""

        for continent in self.iterate_continents():

            continent_name = self.continent_names.get_to(continent)

            if continent_name not in continent_troop_gains:
                raise ValueError(f"Continent {continent} missing from continent_troop_gains")
            else:
                self.continent_troop_gains[continent] = continent_troop_gains[continent_name]

    def is_valid_continent(self, idx: int) -> bool:
        return self.continent_names.to_contains(idx)

    def is_valid_territory(self, idx: int) -> bool:
        return self.territory_names.to_contains(idx)

    def iterate_continents(self) -> Iterable[int]:
        return self.continent_names.iterate_to()

    def iterate_territories(self) -> Iterable[int]:
        return self.territory_names.iterate_to()
