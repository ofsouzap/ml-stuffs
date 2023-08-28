from typing import NamedTuple, Set, List
from random import shuffle
from .world import World


# TODO - implement wildcard territory card


class TerritoryCard(NamedTuple):
    territory: int
    """The territory that the card corresponds to"""
    card_class: int
    """The class of the card. In the base game, there are 3 options for this"""

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, TerritoryCard):
            return self.territory == __value.territory
        else:
            return False


def generate_deck(world: World, card_class_count: int) -> Set[TerritoryCard]:

    territories: List[int] = list(world.iterate_territories())
    card_classes: List[int] = [i % card_class_count for i in range(len(territories))]

    shuffle(card_classes)

    deck: Set[TerritoryCard] = set(
        [TerritoryCard(territories[i], card_classes[i]) for i in range(len(territories))]
    )

    return deck
