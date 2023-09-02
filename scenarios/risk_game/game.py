from typing import Optional, Set, Dict, FrozenSet, Union, List
from random import choice as random_choice
from .world import World
from .setup_board import SetupBoard
from .game_board import GameBoard
from .territory_card import TerritoryCard
from .territory_card import generate_deck as generate_territory_card_deck


class SetupBoardNotFullyOccupiedException(Exception):
    pass


class GameNotSetUpException(Exception):
    pass


class Game:
    """A class for all the details of a running game"""

    DEFAULT_INITIAL_PLACEMENT_ROUND_COUNT: int = 30
    DEFAULT_MAX_ATTACKERS: int = 3
    DEFAULT_MAX_DEFENDERS: int = 2
    DEFAULT_TERRITORY_CARD_CLASS_COUNT: int = 3
    DEFAULT_TERRITORY_CARD_TRADE_IN_OCCUPIED_TERRITORY_BONUS: int = 2
    DEFAULT_ELIMINATING_OPPONENT_TERRITORY_CARD_GAIN_FORCE_TRADE_IN_THRESHOLD: int = 6
    DEFAULT_ELIMINATING_OPPONENT_TERRITORY_CARD_GAIN_FORCE_TRADE_IN_TRADE_UNTIL_BOUND: int = 6

    def __init__(self,
                 world: World,
                 players: Set[int],
                 initial_placement_round_count: Optional[int] = None,
                 max_attackers: Optional[int] = None,
                 max_defenders: Optional[int] = None,
                 territory_card_class_count: Optional[int] = None,
                 territory_card_trade_in_occupied_territory_bonus: Optional[int] = None,
                 elimintating_opponent_territory_card_gain_force_trade_in_threshold: Optional[int] = None,
                 elimintating_opponent_territory_card_gain_force_trade_in_trade_until_bound: Optional[int] = None):

        self._world = world
        self._players = players

        # Game config

        self._initial_placement_round_count: int = initial_placement_round_count or Game.DEFAULT_INITIAL_PLACEMENT_ROUND_COUNT
        self._max_attackers: int = max_attackers or Game.DEFAULT_MAX_ATTACKERS
        self._max_defenders: int = max_defenders or Game.DEFAULT_MAX_DEFENDERS
        self._territory_card_class_count: int = territory_card_class_count or Game.DEFAULT_TERRITORY_CARD_CLASS_COUNT
        self._territory_card_trade_in_occupied_territory_bonus: int = territory_card_trade_in_occupied_territory_bonus or Game.DEFAULT_TERRITORY_CARD_TRADE_IN_OCCUPIED_TERRITORY_BONUS
        self._elimintating_opponent_territory_card_gain_force_trade_in_threshold: int = elimintating_opponent_territory_card_gain_force_trade_in_threshold or Game.DEFAULT_ELIMINATING_OPPONENT_TERRITORY_CARD_GAIN_FORCE_TRADE_IN_THRESHOLD
        self._elimintating_opponent_territory_card_gain_force_trade_in_trade_until_bound: int = elimintating_opponent_territory_card_gain_force_trade_in_trade_until_bound or Game.DEFAULT_ELIMINATING_OPPONENT_TERRITORY_CARD_GAIN_FORCE_TRADE_IN_TRADE_UNTIL_BOUND

        # Set up game

        self.reset_game()

    def reset_game(self) -> None:

        # Boards

        self._setup_board = SetupBoard(self.world)
        self._game_board = None

        # Territory card set-up

        self._territory_card_trade_in_index = 0

        self._player_territory_cards: Dict[int, Set[TerritoryCard]] = {}
        """The territory cards that each player has"""

        for player in self.players:
            self._player_territory_cards[player] = set()

        self._territory_card_deck: Set[TerritoryCard] = generate_territory_card_deck(self.world, self.territory_card_class_count)

    @property
    def players(self) -> Set[int]:
        return self._players

    @property
    def world(self) -> World:
        return self._world

    @property
    def setup_board(self) -> SetupBoard:
        return self._setup_board

    @property
    def game_board(self) -> GameBoard:
        if self._game_board is None:
            raise GameNotSetUpException()
        else:
            return self._game_board

    @property
    def initial_placement_round_count(self) -> int:
        return self._initial_placement_round_count

    @property
    def max_attackers(self) -> int:
        return self._max_attackers

    @property
    def max_defenders(self) -> int:
        return self._max_defenders

    @property
    def territory_card_class_count(self) -> int:
        return self._territory_card_class_count

    @property
    def territory_card_trade_in_occupied_territory_bonus(self) -> int:
        return self._territory_card_trade_in_occupied_territory_bonus

    @property
    def elimintating_opponent_territory_card_gain_force_trade_in_threshold(self) -> int:
        """If eliminating an opponent causes a player to have this many territory cards or more \
then they must trade some in (it is an inclusive bound)"""
        return self._elimintating_opponent_territory_card_gain_force_trade_in_threshold

    @property
    def elimintating_opponent_territory_card_gain_force_trade_in_trade_until_bound(self) -> int:
        """When a player is forced to trade in some territory cards due to gaining them from eliminating an opponent, \
they must keep trading in territory cards until they reach this number of cards or fewer (it is an inclusive bound)"""
        return self._elimintating_opponent_territory_card_gain_force_trade_in_trade_until_bound

    @staticmethod
    def calculate_territory_card_trade_in_troop_gain(trade_in_index: int) -> int:
        """Calculates how many troops a player should gain for trading in a territory card.

Paramters:

    trade_in_index - how many times before this a player has traded in a single set of territory cards. \
On the first territory card trading-in, this should be 0
"""

        if trade_in_index == 0: return 4
        elif trade_in_index == 1: return 6
        elif trade_in_index == 2: return 8
        elif trade_in_index == 3: return 10
        elif trade_in_index == 4: return 12
        elif trade_in_index == 5: return 15
        else: return ((trade_in_index-5) * 5) + 15

    def player_is_active(self, player: int) -> bool:
        """Checks if a player is still active, that is they still own any territories"""
        return len(self.game_board.get_player_territories(player)) > 0

    def generate_game_board(self) -> None:

        if not self._setup_board.is_all_occupied():
            raise SetupBoardNotFullyOccupiedException()
        else:
            self._game_board = GameBoard(self._setup_board)

    def get_player_troop_gain_allowance(self, player: int) -> int:
        """Gets how many troops the player is allowed to place down at the start of their current turn (not including troops from cards)"""

        territories = self.game_board.get_player_territories(player)

        territory_count = len(territories)
        troops_from_territories = territory_count // 3

        full_continents = self.game_board.get_player_full_continents(player)
        troops_from_full_continents = sum(
            map(
                lambda continent: self._world.continent_troop_gains[continent],
                full_continents
            )
        )

        return troops_from_territories + troops_from_full_continents

    def get_winner_player(self) -> Optional[int]:
        """Gets the player who has won or returns None if none have won so far"""

        occupiers = [self.game_board.get_occupier(territory) for territory in self.world.iterate_territories()]

        if all(map(
            lambda x: x == occupiers[0],
            occupiers
        )):

            return occupiers[0]

        else:

            return None

    def player_has_won(self) -> bool:
        """Checks if a player has won by occupying all the territories"""

        return self.get_winner_player() is not None

    def get_player_territory_cards(self, player: int) -> FrozenSet[TerritoryCard]:
        return frozenset(self._player_territory_cards[player])

    def player_has_territory_cards(self, player: int) -> bool:
        """Checks if the player has any territory cards"""
        return len(self.get_player_territory_cards(player)) > 0

    def player_is_able_to_trade_in_territory_cards(self, player: int) -> bool:
        """Checks whether a player has enough cards and the right cards in order to be able to trade in some cards. \
This checks if the player has at least one territory card for every class of territory card \
or they have at least enough of one class of territory card where enough is the number of territory classes in the game"""

        instances_of_classes: List[int] = [0 for _ in range(self.territory_card_class_count)]

        for territory_card in self.get_player_territory_cards(player):
            instances_of_classes[territory_card.card_class] += 1

        has_instance_of_all_classes: bool = all(map(lambda x: x > 0, instances_of_classes))
        has_enough_instances_of_single_class: bool = any(map(lambda x: x >= self.territory_card_class_count, instances_of_classes))

        return has_instance_of_all_classes or has_enough_instances_of_single_class

    def trade_in_player_territory_cards(self, player: int, territory_cards: Union[Set[TerritoryCard], FrozenSet[TerritoryCard]]) -> int:
        """Trades in some of a player's territory cards and returns how many troops they should gain from this.

Parameters:

    player - the player whose territory cards are being traded in

    territory_cards - which of the player's cards are being traded in

Returns:

    troop_gain - the number of troops that the player should be awarded with for trading in these cards
"""

        # Move the cards back into the deck

        for territory_card in territory_cards:

            if territory_card not in self._player_territory_cards[player]:

                raise ValueError("Territory card is not in player's hand")

            else:

                self._player_territory_cards[player].remove(territory_card)
                self._territory_card_deck.add(territory_card)

        # Calculate how many tropops to award the player with

        troop_gain = Game.calculate_territory_card_trade_in_troop_gain(self._territory_card_trade_in_index)
        self._territory_card_trade_in_index += 1

        return troop_gain

    @property
    def territory_card_deck_has_cards_available(self) -> bool:
        """Checks if the territory card deck has any cards available to give to a player"""
        return len(self._territory_card_deck) > 0

    def give_player_random_territory_card(self, player: int) -> TerritoryCard:
        """Gives the player a random new territory card from the deck into their hand and also returns it for reference"""

        assert self.territory_card_deck_has_cards_available, "Deck is empty when trying to give player a card"

        territory_card = random_choice(list(self._territory_card_deck))

        self._territory_card_deck.remove(territory_card)
        self._player_territory_cards[player].add(territory_card)

        return territory_card

    def transfer_player_territory_cards_to_other_player(self, src_player: int, dst_player: int) -> FrozenSet[TerritoryCard]:
        """Transfers all of one player's territory cards to another player and returns the set of transferred cards for reference.

Paramerers:

    src_player - the player to take the cards from

    dst_player - the player to give the cards to

Returns:

    transferred_cards - the set of cards that were transferred. Returned only as a reference
"""

        transferred_cards = self.get_player_territory_cards(src_player)

        self._player_territory_cards[src_player].clear()
        self._player_territory_cards[dst_player].update(transferred_cards)

        return transferred_cards
