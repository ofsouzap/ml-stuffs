from typing import List, Tuple, Iterable, Dict, FrozenSet, Callable, Optional, Set
from collections import defaultdict
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt



class UndirectedGraphBase:
    """Base class for classes representing immutable undirected graphs"""

    @abstractmethod
    def get(self, a: int, b: int) -> bool:
        pass

    def __getitem__(self, key) -> bool:

        # I can't be bothered to check this stuff properly

        assert len(key) == 2

        a = int(key[0])
        b = int(key[1])

        return self.get(a,b)

    @abstractmethod
    def iterate_node_neighbours(self, node: int) -> Iterable[int]:
        pass

    def bfs_find_path(self,
                      start: int,
                      stop_condition: Callable[[int], bool],
                      valid_condition: Optional[Callable[[int], bool]] = None) -> Optional[List[int]]:
        """Performs a breadth-first search on the graph to find a path to a valid node.

Parameters:

    start - the node to start at

    stop_condition - a function that takes a node and returns whether the algorithm should stop when it reaches the node

    valid_condition (optional) - a function that takes a node and returns whether the node is allowed to be on the path. Defaults to always returning True

Returns:

    path - the path from the start node to a node fulfilling the stop condition if one is found, otherwise None
"""

        if valid_condition is None:
            valid_condition = lambda _: True

        visited_nodes: Set[int] = set()
        queue: List[Tuple[int, List[int]]] = []

        queue.append((start, [start]))

        while queue:

            node, node_path = queue.pop(0)
            visited_nodes.add(node)

            if stop_condition(node):
                return node_path

            for other in self.iterate_node_neighbours(node):

                if (other not in visited_nodes) and (valid_condition(other)):

                    other_path = node_path + [other]
                    queue.append((other, other_path))

        # If no path could be found

        return None

    def bfs_find_all(self,
                     start: int,
                     valid_condition: Callable[[int], bool]) -> Iterable[int]:
        """Performs a breadth-first search on the graph to find all nodes reachable from the starting node using a certain subset of the nodes.

Parameters:

    start - the node to start at

    valid_condition - a function that takes a node and returns whether the node is allowed to be on a path to an output node

Returns:

    nodes - the nodes that have a valid path to them
"""

        visited_nodes: Set[int] = set()
        queue: List[Tuple[int, List[int]]] = []

        queue.append((start, [start]))

        while queue:

            node, node_path = queue.pop(0)
            visited_nodes.add(node)

            yield node

            for other in self.iterate_node_neighbours(node):

                if (other not in visited_nodes) and (valid_condition(other)):

                    other_path = node_path + [other]
                    queue.append((other, other_path))


class ArrayUndirectedGraph(UndirectedGraphBase):
    """An immutable undirected graph implementation using an array to store adjacency truth values"""

    def __init__(self,
                 n: int,
                 edges: List[Tuple[int, int]]):

        self._n = n

        data_size: int = ((self._n*self._n)-self._n) // 2
        self._data: npt.NDArray[np.bool_] = np.zeros(shape=(data_size,), dtype=np.bool_)

        for (a,b) in edges:
            if not (0 <= a < self._n):
                raise ValueError(f"The node {a} doesn't exist in the graph")
            elif not (0 <= b < self._n):
                raise ValueError(f"The node {b} doesn't exist in the graph")
            else:
                self._data[self.__index_of(a,b)] = True

    def __index_of(self, a: int, b: int) -> int:

        if a == b:
            raise ValueError("Can't get index of node to itself")
        elif (not (0 <= a < self._n)) or (not (0 <= b < self._n)):
            raise ValueError("Invalid node")

        if a > b:
            tmp = a
            a = b
            b = tmp

        # By here, a < b

        # index = (a/2)*(2n-a-1) + b - a - 1

        return (a*(2*(self._n)-a-1))//2 + b - a - 1

    def get(self, a: int, b: int) -> bool:
        return self._data[self.__index_of(a,b)]

    def iterate_node_neighbours(self, node: int) -> Iterable[int]:
        for other in range(self._n):
            if other == node:
                continue
            else:
                if self.get(node, other):
                    yield other


class AdjacencyListUndirectedGraph(UndirectedGraphBase):
    """An immutable undirected graph implementation using adjacency lists"""

    def __init__(self,
                 n: int,
                 edges: List[Tuple[int, int]]):

        self._n = n
        self._data: Dict[int, FrozenSet[int]] = {}

        # Build adjacency lists

        building_data = defaultdict(lambda: set())

        for (a,b) in edges:
            if not (0 <= a < self._n):
                raise ValueError(f"The node {a} doesn't exist in the graph")
            elif not (0 <= b < self._n):
                raise ValueError(f"The node {b} doesn't exist in the graph")
            else:
                building_data[a].add(b)
                building_data[b].add(a)

        for node in range(self._n):
            node_neighbours = frozenset(building_data[node])
            self._data[node] = node_neighbours

    def get(self, a: int, b: int) -> bool:
        return b in self._data[a]

    def iterate_node_neighbours(self, node: int) -> Iterable[int]:
        return self._data[node]
