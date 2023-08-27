from typing import Optional, Set
from .world import World
from .setup_board import SetupBoard
from .game_board import GameBoard


class SetupBoardNotFullyOccupiedException(Exception):
    pass


class GameNotSetUpException(Exception):
    pass


class Game:
    """A class for all the details of a running game"""

    def __init__(self,
                 world: World,
                 players: Set[int]):

        self._world = world
        self._players = players
        self._setup_board = SetupBoard(world)
        self._game_board: Optional[GameBoard] = None

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
