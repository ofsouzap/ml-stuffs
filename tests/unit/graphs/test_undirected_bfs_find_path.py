from typing import List, Tuple, Callable
import pytest
from graphs import ArrayUndirectedGraph, AdjacencyListUndirectedGraph, UndirectedGraphBase


_CASES: List[Tuple[int,List[Tuple[int,int]],int,Callable[[int],bool],Callable[[int],bool],bool]] = [
    (
        3,
        [
            (0,1),
            (1,0),
            (2,0),
        ],
        1,
        lambda n: n == 0,
        lambda n: True,
        True
    ),
    (
        3,
        [
            (0,1),
            (1,0),
            (2,0),
        ],
        1,
        lambda n: n == 0,
        lambda n: False,
        False
    ),
    (
        2,
        [
            (0,1),
        ],
        0,
        lambda n: n == 1,
        lambda n: True,
        True
    ),
    (
        3,
        [
            (0,1),
        ],
        0,
        lambda n: n == 2,
        lambda n: True,
        False
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
        lambda n: n == 5,
        lambda n: n % 2 == 0,
        False
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
        lambda n: n == 5,
        lambda n: n % 2 == 1,
        True
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
        lambda n: n == 5,
        lambda n: n % 2 == 1,
        True
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
        lambda n: n == 5,
        lambda n: n % 2 == 0,
        False
    ),
]


def run_test(graph: UndirectedGraphBase, start: int, stop_condition: Callable[[int], bool], valid_condition: Callable[[int], bool], exp_res: bool) -> None:

    out = graph.bfs_shortest_path(
        start=start,
        stop_condition=stop_condition,
        valid_condition=valid_condition
    )

    if exp_res:

        assert out is not None, "Path wasn't found but should have been"

        if stop_condition(start):

            assert out == [start], "Path should just be starting node"

        else:

            assert out[0] == start, "Path doesn't start at start"
            assert len(out) > 1, "Path doesn't have at least two nodes"

            for i in range(len(out)-1):

                a = out[i]
                b = out[i+1]

                assert valid_condition(b), "Node in path isn't valid path node"
                assert graph.get(a, b), "Nodes in path aren't neighbours"

    else:
        assert out is None, "Path was found but should not have been"


@pytest.mark.parametrize(("n","edges","start","stop_condition","valid_condition","exp_res"), _CASES)
def test_array(n: int, edges: List[Tuple[int,int]], start: int, stop_condition: Callable[[int], bool], valid_condition: Callable[[int], bool], exp_res: bool) -> None:
    graph = ArrayUndirectedGraph(n, edges)
    run_test(graph, start, stop_condition, valid_condition, exp_res)


@pytest.mark.parametrize(("n","edges","start","stop_condition","valid_condition","exp_res"), _CASES)
def test_adjacency_list(n: int, edges: List[Tuple[int,int]], start: int, stop_condition: Callable[[int], bool], valid_condition: Callable[[int], bool], exp_res: bool) -> None:
    graph = AdjacencyListUndirectedGraph(n, edges)
    run_test(graph, start, stop_condition, valid_condition, exp_res)
