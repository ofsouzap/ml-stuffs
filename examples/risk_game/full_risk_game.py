from scenarios.risk_game import FULL_GAME_WORLD
from scenarios.risk_game.runner import Runner
from scenarios.risk_game.player_controller import RandomizedComputerPlayerController, UncheckedConsolePlayerController


class PrintLogger(Runner.LoggerBase):
    def log(self, message: str):
        pass


world = FULL_GAME_WORLD

player_controllers = [
    RandomizedComputerPlayerController(),
    RandomizedComputerPlayerController(),
    RandomizedComputerPlayerController(),
]

logger = PrintLogger()

runner = Runner(world, player_controllers, logger, initial_placement_round_count=30)

runner.run()

logger.log(f"Winner is player {runner.winner}")
