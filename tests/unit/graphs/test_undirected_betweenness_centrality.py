from typing import List, Tuple, Callable, Set, Dict
import pytest
from graphs import ArrayUndirectedGraph, AdjacencyListUndirectedGraph, UndirectedGraphBase


_CASES: List[Tuple[int,List[Tuple[int,int]],Dict[int,float]]] = [
    (
        3,
        [
            (0,1),
            (1,2),
        ],
        {
            0: 0.0,
            1: 1.0,
            2: 0.0,
        },
    ),
    (
        3,
        [
            (0,1),
            (1,2),
            (2,0),
        ],
        {
            0: 0.0,
            1: 0.0,
            2: 0.0,
        },
    ),
    (
        2,
        [
            (0,1),
        ],
        {
            0: 0.0,
            1: 0.0,
        },
    ),
    (
        4,
        [
            (0,1),
            (1,2),
        ],
        {
            0: 0.0,
            1: 1.0,
            2: 0.0,
        },
    ),
    (
        5,
        [
            (0,1),
            (1,2),
            (1,3),
            (3,4),
        ],
        {
            0: 0.0,
            1: 5.0,
            2: 0.0,
            3: 3.0,
            4: 0.0,
        },
    ),
    (
        4,
        [
            (0,1),
            (0,2),
            (3,1),
            (3,2),
        ],
        {
            0: 0.5,
            1: 0.5,
            2: 0.5,
            3: 0.5,
        },
    ),
]


def run_test(graph: UndirectedGraphBase, exps: Dict[int, float]) -> None:

    for node in exps:
        out = graph.get_betweenness_centrality(node)
        exp = exps[node]
        assert out == exp


@pytest.mark.parametrize(("n","edges","exps"), _CASES)
def test_array(n: int, edges: List[Tuple[int,int]], exps: Dict[int, float]) -> None:
    graph = ArrayUndirectedGraph(n, edges, precalculate_betweenness_centrality=True)
    run_test(graph, exps)


@pytest.mark.parametrize(("n","edges","exps"), _CASES)
def test_adjacency_list(n: int, edges: List[Tuple[int,int]], exps: Dict[int, float]) -> None:
    graph = AdjacencyListUndirectedGraph(n, edges, precalculate_betweenness_centrality=True)
    run_test(graph, exps)
