from typing import Iterable, NamedTuple, Optional, DefaultDict, Set, List
from abc import ABC, abstractmethod
from collections import defaultdict
from random import choice as random_choice
from random import sample as random_sample
from random import randint
from scenarios.risk_game.game import Game
from scenarios.risk_game.territory_card import TerritoryCard
from .game import Game
from .territory_card import TerritoryCard
from _debug_timer import CumulativeTimer


class AttackAction(NamedTuple):
    from_territory: int
    to_territory: int
    attackers: int


class TroopRelocateAction(NamedTuple):
    from_territory: int
    to_territory: int
    troop_count: int


class PlayerControllerBase(ABC):

    def __init__(self):

        self._self_player: Optional[int] = None

    @property
    def self_player(self) -> int:
        if self._self_player is not None:
            return self._self_player
        else:
            raise Exception("Self player hasn't been set")

    def set_self_player(self, self_player: int) -> None:
        self._self_player = self_player

    @abstractmethod
    def decide_initial_placing_board_occupy_territory(self, game: Game) -> int:
        """Tells the player to choose a single unoccupied territory to occupy during the initial placement phase.

Parameters:

    game - the instance of the running game

Returns:

    territory - the territory the player has chosen to place a troop in to occupy
"""
        pass

    @abstractmethod
    def decide_initial_placing_troop_placement_territory(self, game: Game) -> int:
        """Tells the player to choose a single territory to place another troop in during the initial placement phase.

Parameters:

    game - the instance of the running game

Returns:

    territory - the territory the player has chosen to place a troop in
"""
        pass

    @abstractmethod
    def decide_trade_in_territory_cards(self, game: Game) -> Optional[Set[TerritoryCard]]:
        """Tells the player to choose what territory cards they want to trade in or if they don't want to trade any in.

Parameters:

    game - the instance of the running game

Returns:

    cards_to_trade_in - the cards the player has chosen to spend or None if they decided not to spend. \
The set of cards chosen must have the right number of cards and follow the validity rules of cards to trade in together.
"""
        pass

    @abstractmethod
    def decide_forced_trade_in_territory_cards(self, game: Game) -> Set[TerritoryCard]:
        """Tells the player to choose what territory cards they want to trade in which they must choose, they cannot choose not to trade in cards.

Parameters:

    game - the instance of the running game

Returns:

    cards_to_trade_in - the cards the player has chosen to spend. \
The set of cards chosen must have the right number of cards and follow the validity rules of cards to trade in together.
"""
        pass

    @abstractmethod
    def decide_troop_placement_territories(self, game: Game, troop_count: int) -> Iterable[int]:
        """Tells the player to choose where to place the given number of troops.

Parameters:

    game - the instance of the running game

    troop_count - the number of troops the player must place

Returns:

    territories - an iterable of the territories the player has chosen to place their troops in. \
The length of `territories` will be `troop_count` and repeated values are allowed
"""
        pass

    @abstractmethod
    def decide_attack_action(self, game: Game) -> Optional[AttackAction]:
        """Tells the player to choose where they want to attack next or if they want to stop attacking now.

Parameters:

    game - the instance of the running game

Returns:

    attack_action - the attacking action the player has chosen to perform. \
If None then this means that the player has chosen not to relocate
"""
        pass

    @abstractmethod
    def decide_defender_count(self, game: Game, attack_action: AttackAction) -> int:
        """Tells the player to choose how many defenders they want to use to defend against an attack.

Parameters:

    game - the instance of the running game

    attack_action - the attack that the player must defend against

Returns:

    defenders - the number of defenders the player has chosen to use
"""
        pass

    @abstractmethod
    def decide_troop_relocate(self, game: Game) -> Optional[TroopRelocateAction]:
        """Tells the player to choose where they want to relocate troops from and to or if they don't want to relocate any troops.

Parameters:

    game - the instance of the running game

Returns:

    troop_relocate_action - the relocation action the player has chosen to perform. \
If None then this means that the player has chosen not to relocate
"""
        pass


class RandomizedComputerPlayerController(PlayerControllerBase):

    def decide_initial_placing_board_occupy_territory(self, game: Game) -> int:
        with CumulativeTimer("decide_initial_placing_board_occupy_territory"):
            options = []

            for territory in game.world.iterate_territories():
                if game.setup_board.get_occupant(territory) is None:
                    options.append(territory)

            return random_choice(options)

    def decide_initial_placing_troop_placement_territory(self, game: Game) -> int:
        with CumulativeTimer("decide_initial_placing_troop_placement_territory"):
            options = game.game_board.get_player_territories(self.self_player)

            return random_choice(list(options))

    def decide_trade_in_territory_cards(self, game: Game) -> Optional[Set[TerritoryCard]]:
        with CumulativeTimer("decide_trade_in_territory_cards"):
            options_by_class: List[Set[TerritoryCard]] = [set() for _ in range(game.territory_card_class_count)]

            for territory_card in game.get_player_territory_cards(self.self_player):
                options_by_class[territory_card.card_class].add(territory_card)

            has_instance_of_all_classes: bool = all(map(lambda x: len(x) > 0, options_by_class))
            has_enough_instances_of_single_class: bool = any(map(lambda x: len(x) >= game.territory_card_class_count, options_by_class))

            if has_enough_instances_of_single_class or has_instance_of_all_classes:

                if randint(0, max([len(xs) for xs in options_by_class])) == 0:

                    # Choosing not to trade in any cards

                    return None

                else:

                    use_instance_of_all_classes: bool

                    if has_enough_instances_of_single_class != has_instance_of_all_classes:
                        use_instance_of_all_classes = has_instance_of_all_classes
                    elif has_enough_instances_of_single_class and has_instance_of_all_classes:
                        use_instance_of_all_classes = randint(0,1) == 0
                    else:
                        raise Exception("This shouldn't occur")

                    if use_instance_of_all_classes:

                        cards_to_trade_in: Set[TerritoryCard] = set([random_choice(list(xs)) for xs in options_by_class])

                        return cards_to_trade_in

                    else:

                        options_choices = list(filter(
                            lambda xs: len(xs) >= game.territory_card_class_count,
                            options_by_class
                        ))

                        options_chosen = random_choice(options_choices)

                        cards_to_trade_in: Set[TerritoryCard] = set(random_sample(options_chosen, k=game.territory_card_class_count))

                        return cards_to_trade_in

            else:

                return None

    def decide_forced_trade_in_territory_cards(self, game: Game) -> Set[TerritoryCard]:
        with CumulativeTimer("decide_forced_trade_in_territory_cards"):
            options_by_class: List[Set[TerritoryCard]] = [set() for _ in range(game.territory_card_class_count)]

            for territory_card in game.get_player_territory_cards(self.self_player):
                options_by_class[territory_card.card_class].add(territory_card)

            has_instance_of_all_classes: bool = all(map(lambda x: len(x) > 0, options_by_class))
            has_enough_instances_of_single_class: bool = any(map(lambda x: len(x) >= game.territory_card_class_count, options_by_class))

            assert has_instance_of_all_classes or has_enough_instances_of_single_class, "Trying to force player to choose to trade in cards when they can't"

            use_instance_of_all_classes: bool

            if has_enough_instances_of_single_class != has_instance_of_all_classes:
                use_instance_of_all_classes = has_instance_of_all_classes
            elif has_enough_instances_of_single_class and has_instance_of_all_classes:
                use_instance_of_all_classes = randint(0,1) == 0
            else:
                raise Exception("This shouldn't occur")

            if use_instance_of_all_classes:

                cards_to_trade_in: Set[TerritoryCard] = set([random_choice(list(xs)) for xs in options_by_class])

                return cards_to_trade_in

            else:

                options_choices = list(filter(
                    lambda xs: len(xs) >= game.territory_card_class_count,
                    options_by_class
                ))

                options_chosen = random_choice(options_choices)

                cards_to_trade_in: Set[TerritoryCard] = set(random_sample(options_chosen, k=game.territory_card_class_count))

                return cards_to_trade_in

    def decide_troop_placement_territories(self, game: Game, troop_count: int) -> Iterable[int]:
        with CumulativeTimer("decide_troop_placement_territories"):
            options = game.game_board.get_player_territories(self.self_player)
            return [random_choice(list(options)) for _ in range(troop_count)]

    def decide_attack_action(self, game: Game) -> Optional[AttackAction]:
        with CumulativeTimer("decide_attack_action"):
            # Find options

            options: DefaultDict[int, Set[int]] = defaultdict(lambda: set())

            for from_territory in game.game_board.get_player_territories(self.self_player):

                if game.game_board.get_troop_count(from_territory) <= 1:
                    continue

                for to_territory in game.world.iterate_territory_neighbours(from_territory):

                    if game.game_board.get_occupier(to_territory) != self.self_player:
                        options[from_territory].add(to_territory)

            # Choose option

            if randint(0,len(options)) == 0:

                return None

            else:

                from_territory = random_choice(list(options.keys()))

                to_options = options[from_territory]
                if len(to_options) > 0:

                    to_territory = random_choice(list(to_options))

                    available_attackers = game.game_board.get_troop_count(from_territory) - 1
                    max_attackers = min(available_attackers, game.max_attackers)
                    attackers = randint(1,max_attackers)

                    return AttackAction(from_territory, to_territory, attackers)

                else:
                    return None

    def decide_defender_count(self, game: Game, attack_action: AttackAction) -> int:

        territory_troops = game.game_board.get_troop_count(attack_action.to_territory)

        return min(game.max_defenders, territory_troops)

    def decide_troop_relocate(self, game: Game) -> Optional[TroopRelocateAction]:
        with CumulativeTimer("decide_troop_relocate"):
            # Find from options

            from_options: List[int] = []

            for territory in game.game_board.get_player_territories(self.self_player):
                if game.game_board.get_troop_count(territory) > 1:
                    from_options.append(territory)

            # Decide if relocating

            if randint(0,len(from_options)) == 0:

                return None

            else:

                from_territory = random_choice(from_options)

                # Find to options

                to_options: List[int] = []

                for territory in game.world.graph.bfs_find_all(
                    start=from_territory,
                    valid_condition=lambda n: (n != from_territory) and (game.game_board.get_occupier(n) == self.self_player)
                ):
                    if territory != from_territory:
                        to_options.append(territory)

                # Choose to option

                if len(to_options) == 0:

                    return None

                else:

                    to_territory = random_choice(to_options)

                    from_territory_troops = game.game_board.get_troop_count(from_territory)

                    if from_territory_troops <= 1:
                        return None

                    # Choose troops moved count

                    troops_moved_count = randint(1, from_territory_troops-1)

                    return TroopRelocateAction(from_territory, to_territory, troops_moved_count)


class AggressiveRandomComputerPlayerContoller(RandomizedComputerPlayerController):
    """A variation of the random computer player controller that will be more aggressive in order to make simulated games end sooner"""

    def decide_attack_action(self, game: Game) -> Optional[AttackAction]:
        with CumulativeTimer("decide_attack_action"):
            # Find options

            options: DefaultDict[int, Set[int]] = defaultdict(lambda: set())

            for from_territory in game.game_board.get_player_territories(self.self_player):

                if game.game_board.get_troop_count(from_territory) <= 1:
                    continue

                for to_territory in game.world.iterate_territory_neighbours(from_territory):

                    if game.game_board.get_occupier(to_territory) != self.self_player:
                        options[from_territory].add(to_territory)

            # Choose option

            if len(options) == 0:

                # If can't attack
                return None

            else:

                # Attack if can

                from_territory = random_choice(list(options.keys()))

                to_options = options[from_territory]
                if len(to_options) > 0:

                    to_territory = random_choice(list(to_options))

                    available_attackers = game.game_board.get_troop_count(from_territory) - 1
                    max_attackers = min(available_attackers, game.max_attackers)

                    return AttackAction(from_territory, to_territory, max_attackers)

                else:
                    return None

    def decide_defender_count(self, game: Game, attack_action: AttackAction) -> int:
        return 1  # Always defend with the least number of troops possible


class UncheckedConsolePlayerController(PlayerControllerBase):
    """A very basic player controller for a player playing through the console. Only for use during debugging because it is really bad"""

    def __print_setup_board_state(self, game: Game) -> None:
        print(f"""\
+--------------------+
| Set-Up Board State |
+--------------------+

Territories:
\t""" + "\n\t".join([f"{territory} ({game.world.get_territory_name(territory)}) - " + \
    (f"{game.setup_board.get_occupant(territory)}'s troops" if game.setup_board.get_occupant(territory) is not None else "empty") \
    for territory in game.world.iterate_territories()]) + """
""")

    def __print_game_board_state(self, game: Game) -> None:
        print(f"""\
+------------+
| Game State |
+------------+

Players:
\t""" + "\n\t".join([str(player) for player in game.players]) + """

Your Cards:
\t""" + "\n\t".join([f"{game.world.get_territory_name(card.territory)}-{card.card_class}" \
    for card in sorted(game.get_player_territory_cards(self.self_player), key=lambda card: card.territory)]) + """

Territories:
\t""" + "\n\t".join([f"{territory} ({game.world.get_territory_name(territory)}) - {game.game_board.get_troop_count(territory)} of player {game.game_board.get_occupier(territory)}'s troops"
    for territory in game.world.iterate_territories()]) + """
""")

    def decide_initial_placing_board_occupy_territory(self, game: Game) -> int:
        self.__print_setup_board_state(game)
        return int(input("Choose which territory to occupy> "))

    def decide_initial_placing_troop_placement_territory(self, game: Game) -> int:
        self.__print_game_board_state(game)
        return int(input("Choose where to put a reinforcement troop> "))

    def decide_trade_in_territory_cards(self, game: Game) -> Optional[Set[TerritoryCard]]:

        if input("Do you want to attack? ").lower() in ["y", "yes", "1", "true"]:
            return self.decide_forced_trade_in_territory_cards(game)
        else:
            return None

    def decide_forced_trade_in_territory_cards(self, game: Game) -> Set[TerritoryCard]:

        self.__print_game_board_state(game)

        territories: Set[int] = set(
            map(
                lambda x: game.world.territory_by_name(x.strip()),
                input("Which territories' territory cards will you trade in (CSV)> ").split(",")
            )
        )

        cards_to_trade_in: Set[TerritoryCard] = set()

        for card in game.get_player_territory_cards(self.self_player):
            if card.territory in territories:
                cards_to_trade_in.add(card)

        return cards_to_trade_in

    def decide_troop_placement_territories(self, game: Game, troop_count: int) -> Iterable[int]:
        if troop_count == 0:
            return []
        self.__print_game_board_state(game)
        return map(lambda x: int(x.strip()), input(f"Choose where to deploy {troop_count} troops (CSV)> ").split(","))

    def decide_attack_action(self, game: Game) -> Optional[AttackAction]:
        self.__print_game_board_state(game)
        if input("Do you want to attack? ").lower() in ["y", "yes", "1", "true"]:
            return AttackAction(
                from_territory=int(input("Where from> ")),
                to_territory=int(input("Where to> ")),
                attackers=int(input("How many attackers> "))
            )
        else:
            return None

    def decide_defender_count(self, game: Game, attack_action: AttackAction) -> int:
        self.__print_game_board_state(game)
        return int(input(f"""You are being attacked from {game.world.get_territory_name(attack_action.from_territory)} to {game.world.get_territory_name(attack_action.to_territory)} with {attack_action.attackers} troops.
How many troops to defend with> """))

    def decide_troop_relocate(self, game: Game) -> Optional[TroopRelocateAction]:
        self.__print_game_board_state(game)
        if input("Do you want to relocate any troops? ").lower() in ["y", "yes", "1", "true"]:
            return TroopRelocateAction(
                from_territory=int(input("Where from> ")),
                to_territory=int(input("Where to> ")),
                troop_count=int(input("How many troops> "))
            )
        else:
            return None
