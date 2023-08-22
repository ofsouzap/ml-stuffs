from typing import Dict, TypeVar, Generic, Iterator


A = TypeVar("A")
B = TypeVar("B")


class BijMap(Generic[A, B]):
    """A class for a bijective mapping"""

    def __init__(self):
        self.__to: Dict[A, B] = {}
        self.__from: Dict[B, A] = {}

    def __add(self, a: A, b: B) -> None:
        self.__to[a] = b
        self.__from[b] = a

    def set_to(self, a: A, b: B) -> None:
        self.__add(a, b)

    def set_from(self, b: B, a: A) -> None:
        self.__add(a, b)

    def get_to(self, a: A) -> B:
        if a in self.__to:
            return self.__to[a]
        else:
            raise KeyError(a)

    def get_from(self, b: B) -> A:
        if b in self.__from:
            return self.__from[b]
        else:
            raise KeyError(b)

    def to_contains(self, a: A) -> bool:
        return a in self.__to

    def from_contins(self, b: B) -> bool:
        return b in self.__from

    def iterate_to(self) -> Iterator[A]:
        return iter(self.__to.keys())

    def iterate_from(self) -> Iterator[B]:
        return iter(self.__from.keys())

    @property
    def size(self) -> int:
        return len(self.__to)
