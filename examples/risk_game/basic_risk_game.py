from scenarios.risk_game.world import World
from scenarios.risk_game.runner import Runner
from scenarios.risk_game.player_controller import RandomizedComputerPlayerController, UncheckedConsolePlayerController


class PrintLogger(Runner.LoggerBase):
    def log(self, message: str):
        print(f"INFO: {message}")


world = World(
    continent_territories={
        "C1": ["C1a", "C1b"],
        "C2": ["C2a", "C2b", "C2c"]
    },
    continent_troop_gains={
        "C1": 2,
        "C2": 3
    },
    territory_connections={
        "C1a": ["C1b"],
        "C1b": ["C1a", "C2a"],
        "C2a": ["C2b", "C2c"],
        "C2b": ["C2a", "C2c"],
        "C2c": ["C2a", "C2b"],
    }
)

player_controllers = [
    UncheckedConsolePlayerController(0),
    RandomizedComputerPlayerController(1),
]

logger = PrintLogger()

runner = Runner(world, player_controllers, logger, initial_placement_round_count=5)

runner.run()

logger.log(f"Winner is player {runner.winner}")