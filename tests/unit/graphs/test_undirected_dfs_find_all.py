from typing import List, Tuple, Callable, Set
import pytest
from graphs import ArrayUndirectedGraph, AdjacencyListUndirectedGraph, UndirectedGraphBase


_CASES: List[Tuple[int,List[Tuple[int,int]],int,Callable[[int],bool],Set[int]]] = [
    (
        3,
        [
            (0,1),
            (1,0),
            (2,0),
        ],
        1,
        lambda n: True,
        {0,1,2}
    ),
    (
        3,
        [
            (0,1),
            (1,0),
            (2,0),
        ],
        1,
        lambda n: False,
        {1}
    ),
    (
        2,
        [
            (0,1),
        ],
        0,
        lambda n: True,
        {0,1}
    ),
    (
        3,
        [
            (0,1),
        ],
        0,
        lambda n: True,
        {0,1}
    ),
    (
        6,
        [
            (0,2),
            (1,3),
            (2,4),
            (3,5),
        ],
        0,
        lambda n: n % 2 == 0,
        {0,2,4}
    ),
    (
        6,
        [
            (0,2),
            (1,3),
            (2,4),
            (3,5),
        ],
        1,
        lambda n: n % 2 == 1,
        {1,3,5}
    ),
    (
        6,
        [
            (0,1),
            (0,2),
            (1,3),
            (2,4),
            (3,5),
        ],
        0,
        lambda n: n % 2 == 1,
        {0,1,3,5}
    ),
    (
        6,
        [
            (0,1),
            (0,2),
            (0,3),
            (0,4),
            (3,5),
        ],
        0,
        lambda n: n % 2 == 0,
        {0,2,4}
    ),
]


def run_test(graph: UndirectedGraphBase, start: int, valid_condition: Callable[[int], bool], exp: Set[int]) -> None:

    out = set(graph.dfs_find_all(
        start=start,
        valid_condition=valid_condition
    ))

    assert out == exp


@pytest.mark.parametrize(("n","edges","start","valid_condition","exp"), _CASES)
def test_array(n: int, edges: List[Tuple[int,int]], start: int, valid_condition: Callable[[int], bool], exp: Set[int]) -> None:
    graph = ArrayUndirectedGraph(n, edges)
    run_test(graph, start, valid_condition, exp)


@pytest.mark.parametrize(("n","edges","start","valid_condition","exp"), _CASES)
def test_adjacency_list(n: int, edges: List[Tuple[int,int]], start: int, valid_condition: Callable[[int], bool], exp: Set[int]) -> None:
    graph = AdjacencyListUndirectedGraph(n, edges)
    run_test(graph, start, valid_condition, exp)
