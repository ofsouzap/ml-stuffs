from typing import Iterable, List, Tuple, Optional
from abc import ABC, abstractmethod
from random import shuffle
from .player_controller import PlayerControllerBase
from .game import Game
from .world import World
from .battle import battle


class InvalidPlayerDecisionException(Exception):
    pass


class Runner:

    class LoggerBase(ABC):
        @abstractmethod
        def log(self, message: str):
            raise NotImplementedError()

    class __EmptyLogger(LoggerBase):
        def log(self, message: str):
            pass

    def __init__(self,
                 world: World,
                 player_controllers: Iterable[PlayerControllerBase],
                 logger: Optional[LoggerBase] = None,
                 initial_placement_round_count: Optional[int] = None,
                 max_attackers: Optional[int] = None,
                 max_defenders: Optional[int] = None):

        self._player_controllers: List[PlayerControllerBase] = list(player_controllers)
        self._game: Game = Game(
            world=world,
            players=set(range(len(self._player_controllers))),
            initial_placement_round_count=initial_placement_round_count,
            max_attackers=max_attackers,
            max_defenders=max_defenders,
        )

        self._running = False
        self._has_run = False
        self._winner: Optional[int] = None

        self._logger: Runner.LoggerBase

        if logger is not None:
            self._logger = logger
        else:
            self._logger = Runner.__EmptyLogger()

    def log(self, *args, **kwargs):
        return self._logger.log(*args, **kwargs)

    @property
    def game(self) -> Game:
        return self._game

    @property
    def world(self) -> World:
        return self.game.world

    @property
    def winner(self) -> Optional[int]:
        return self._winner

    def run(self) -> None:

        if self._has_run or self._running:
            raise Exception("Can't start runner running when it has already run or is still running")

        self._running = True

        # Decide player order

        player_order: List[int] = self._generate_player_order()

        self.log("Player order: " + ", ".join([str(player) for player in player_order]))

        # Initial Placement stage

        self._run_initial_placement(player_order)

        # Main Game stage

        self._run_main_game(player_order)

        # Once game finished

        self._running = False
        self._has_run = True

    def _generate_player_order(self) -> List[int]:
        players = list(range(len(self._player_controllers)))
        shuffle(players)
        return players

    def _run_initial_placement(self, player_order: List[int]) -> None:

        player_idx: int = 0
        initial_placement_round_idx: int = 0

        # First stage of occupying all the territories

        player_idx, initial_placement_round_idx = self._run_initial_placement_setup_board(
            player_order,
            start_player_idx=player_idx,
            start_initial_placement_round_idx=initial_placement_round_idx
        )

        # Generate the game board once all territories are occupied

        self.game.generate_game_board()

        # Second stage of reinforcing the already-occupied territories

        player_idx, initial_placement_round_idx = self._run_initial_placement_game_board(
            player_order,
            start_player_idx=player_idx,
            start_initial_placement_round_idx=initial_placement_round_idx,
            initial_placement_round_count=self.game.initial_placement_round_count
        )

        # Some checks

        assert player_idx == 0
        assert initial_placement_round_idx == self.game.initial_placement_round_count

    def _run_initial_placement_setup_board(self,
                                          player_order: List[int],
                                          start_player_idx: int,
                                          start_initial_placement_round_idx: int) -> Tuple[int, int]:

        player_idx: int = start_player_idx
        initial_placement_round_idx: int = start_initial_placement_round_idx

        while not self.game._setup_board.is_all_occupied():

            player = player_order[player_idx]

            occupy_territory = self._player_controllers[player].decide_initial_placing_board_occupy_territory(self.game)

            if self.game.setup_board.get_occupant(occupy_territory) is not None:
                raise InvalidPlayerDecisionException("Trying to occupy already-occupied territory")

            self.game._setup_board.set_occupant(occupy_territory, player)

            self.log(f"Player {player} occupys {self.world.territory_names.get_to(occupy_territory)}")

            player_idx += 1
            if player_idx == len(player_order):
                player_idx = 0
                initial_placement_round_idx += 1

        return player_idx, initial_placement_round_idx

    def _run_initial_placement_game_board(self,
                                         player_order: List[int],
                                         start_player_idx: int,
                                         start_initial_placement_round_idx: int,
                                         initial_placement_round_count: int) -> Tuple[int, int]:

        player_idx: int = start_player_idx
        initial_placement_round_idx: int = start_initial_placement_round_idx

        while initial_placement_round_idx < initial_placement_round_count:

            player = player_order[player_idx]

            place_territory = self._player_controllers[player].decide_initial_placing_troop_placement_territory(self.game)

            if self.game.game_board.get_occupier(place_territory) != player:
                raise InvalidPlayerDecisionException("Trying to reinforce a territory that isn't the player's")

            self.game.game_board.add_troops(place_territory, 1)
            self.log(f"Player {player} reinforces {self.world.territory_names.get_to(place_territory)} with a troop")

            player_idx += 1
            if player_idx == len(player_order):
                player_idx = 0
                initial_placement_round_idx += 1

        return player_idx, initial_placement_round_idx

    def _run_main_game(self, player_order: List[int]) -> None:

        round_idx: int = 0
        player_idx: int = 0

        while not self.game.player_has_won():

            player = player_order[player_idx]

            self._run_main_game_player_turn(player)

            player_idx += 1
            if player_idx == len(player_order):
                player_idx = 0
                round_idx += 1

        self._winner = self.game.get_winner_player()
        assert self._winner is not None

    def _run_main_game_player_turn(self, player: int) -> None:

        self._run_main_game_player_turn_placement_phase(player)

        if not self.game.player_has_won():
            self._run_main_game_player_turn_attack_phase(player)

        if not self.game.player_has_won():
            self._run_main_game_player_turn_relocation_phase(player)

    def _run_main_game_player_turn_placement_phase(self, player: int) -> None:

        troops_to_place = self.game.get_player_troop_gain_allowance(player)

        # TODO - implement cards to gain more troops

        # Have player make placing location decision

        player_controller = self._player_controllers[player]

        placing_territories = list(player_controller.decide_troop_placement_territories(self.game, troops_to_place))

        # Check all choices are valid

        if len(placing_territories) != troops_to_place:
            raise InvalidPlayerDecisionException("Placing incorrect number of troops")

        for territory in placing_territories:
            if self.game.game_board.get_occupier(territory) != player:
                raise InvalidPlayerDecisionException()

        # Place troops

        for territory in placing_territories:
            self.game.game_board.add_troops(territory, 1)
            self.log(f"{player} adds a troop to {self.world.territory_names.get_to(territory)}")

    def _run_main_game_player_turn_attack_phase(self, player: int) -> None:

        player_controller = self._player_controllers[player]

        while (not self.game.player_has_won()) and (attack_action := player_controller.decide_attack_action(self.game)):

            # Check attack is valid

            if self.game.game_board.get_occupier(attack_action.from_territory) != player:
                raise InvalidPlayerDecisionException("Trying to attack from another player's territory")

            if self.game.game_board.get_occupier(attack_action.to_territory) == player:
                raise InvalidPlayerDecisionException("Player can't attack themselves")

            if attack_action.attackers < 1:
                raise InvalidPlayerDecisionException("Number of attackers must be positive")

            if attack_action.attackers > self.game.max_attackers:
                raise InvalidPlayerDecisionException("Number of attackers is greater than the maximum allowed")

            if self.game.game_board.get_troop_count(attack_action.from_territory) - attack_action.attackers < 1:
                raise InvalidPlayerDecisionException("Trying to attack with more troops than player has spare from selected territory")

            if not self.world.territories_are_neighbours(attack_action.from_territory, attack_action.to_territory):
                raise InvalidPlayerDecisionException("Territory being attacked from and territory attacking into must be neighbours")

            # Get defender count

            defender = self.game.game_board.get_occupier(attack_action.to_territory)

            self.log(f"Player {player} will \
attack player {defender} \
at {self.world.territory_names.get_to(attack_action.to_territory)} \
from {self.world.territory_names.get_to(attack_action.from_territory)} \
with {attack_action.attackers} troops")

            defender_controller = self._player_controllers[defender]
            defender_count = defender_controller.decide_defender_count(self.game, attack_action)

            # Check defender count

            if defender_count < 1:
                raise InvalidPlayerDecisionException("Number of defenders must be positive")

            if defender_count > self.game.max_defenders:
                raise InvalidPlayerDecisionException("Number of defenders is greater than the maximum allowed")

            # Do battle

            self.log(f"Player {defender} is defending with {defender_count} troops")

            killed_attackers, killed_defenders = battle(
                attackers=attack_action.attackers,
                defenders=defender_count
            )

            surviving_attackers = attack_action.attackers - killed_attackers

            self.log(f"{killed_attackers} attackers were killed and {killed_defenders} defenders were killed")

            # Kill troops killed

            self.game.game_board.remove_troops(attack_action.from_territory, killed_attackers)
            self.game.game_board.remove_troops(attack_action.to_territory, killed_defenders)

            # Occupy the territory if left empty

            if self.game.game_board.get_troop_count(attack_action.to_territory) == 0:

                # Occupy territory

                self.game.game_board.remove_troops(attack_action.from_territory, surviving_attackers)
                self.game.game_board.set_occupation(attack_action.to_territory, player, surviving_attackers)

                self.log(f"Player {player} has occupied {self.world.territory_names.get_to(attack_action.to_territory)} with {surviving_attackers} troops")

    def _run_main_game_player_turn_relocation_phase(self, player: int) -> None:

        player_controller = self._player_controllers[player]

        relocate_action = player_controller.decide_troop_relocate(self.game)

        if relocate_action is None:

            self.log(f"Player {player} chose not to relocate any troops")

        else:

            # Check values

            if self.game.game_board.get_occupier(relocate_action.from_territory) != player:
                raise InvalidPlayerDecisionException("Player trying to relocate from another player's territory")

            if self.game.game_board.get_occupier(relocate_action.to_territory) != player:
                raise InvalidPlayerDecisionException("Player trying to relocate to another player's territory")

            if relocate_action.to_territory == relocate_action.from_territory:
                raise InvalidPlayerDecisionException("Trying to relocate from a territory to itself")

            if relocate_action.troop_count <= 0:
                raise InvalidPlayerDecisionException("Number of troops relocated must be positive")

            if self.game.game_board.get_troop_count(relocate_action.from_territory) - relocate_action.troop_count < 1:
                raise InvalidPlayerDecisionException("Relocation chosen would leave no troops in territory")

            if not self.game.game_board.territories_have_continuous_route_with_same_occupier(relocate_action.from_territory, relocate_action.to_territory):
                raise InvalidPlayerDecisionException("Relocation territories aren't connected by only player's territories")

            # Move troops

            self.game.game_board.remove_troops(relocate_action.from_territory, relocate_action.troop_count)
            self.game.game_board.add_troops(relocate_action.to_territory, relocate_action.troop_count)

            self.log(f"Player {player} has relocated \
{relocate_action.troop_count} troops \
from {self.world.territory_names.get_to(relocate_action.from_territory)} \
to {self.world.territory_names.get_to(relocate_action.to_territory)}")
