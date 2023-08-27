from typing import Optional
from .world import World
from .setup_board import SetupBoard
from .game_board import GameBoard


class GameNotSetUpException(Exception):
    pass


class Game:
    """A class for all the details of a running game"""

    def __init__(self,
                 world: World):

        self.world = world
        self._setup_board = SetupBoard(world)
        self._game_board: Optional[GameBoard] = None

    def get_player_troop_gain_allowance(self, player: int) -> int:
        """Gets how many troops the player is allowed to place down at the start of their current turn (not including troops from cards)"""

        if self._game_board is None:
            raise GameNotSetUpException()

        territories = self._game_board.get_player_territories(player)

        territory_count = len(territories)
        troops_from_territories = territory_count // 3

        full_continents = self._game_board.get_player_full_continents(player)
        troops_from_full_continents = sum(
            map(
                lambda continent: self.world.continent_troop_gains[continent],
                full_continents
            )
        )

        return troops_from_territories + troops_from_full_continents
