from typing import List, Tuple
import pytest
from graphs import UndirectedGraph


_CASES: List[Tuple[int,List[Tuple[int,int]]]] = [
    (
        3,
        [
            (0,1),
            (1,0),
            (2,0),
        ]
    ),
    (
        2,
        [
            (0,1),
        ]
    ),
    (
        3,
        [
            (0,1),
        ]
    ),
    (
        5,
        [
            (0,1),
            (3,2),
            (3,2),
            (3,2),
            (2,4),
            (4,3),
        ]
    ),
    (
        4,
        [
            (0,1),
            (0,2),
            (0,3),
            (1,2),
            (1,3),
            (2,3),
        ]
    ),
    (
        4,
        []
    ),
]


@pytest.mark.parametrize(("n","edges"), _CASES)
def test_init(n: int, edges: List[Tuple[int,int]]) -> None:
    graph = UndirectedGraph(n, edges)


@pytest.mark.parametrize(("n","edges"), _CASES)
def test_get(n: int, edges: List[Tuple[int,int]]) -> None:

    graph = UndirectedGraph(n, edges)

    for a in range(n):
        for b in range(n):

            if a == b:
                continue

            exp = ((a,b) in edges) or ((b,a) in edges)
            res_index = graph[a,b]
            res_get = graph.get(a,b)
            assert res_index == res_get == exp
