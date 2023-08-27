from scenarios.risk_game.world import World
from scenarios.risk_game.runner import Runner
from scenarios.risk_game.player_controller import RandomizedComputerPlayerController, UncheckedConsolePlayerController


class PrintLogger(Runner.LoggerBase):
    def log(self, message: str):
        print(f"INFO: {message}")


world = World(
    continent_territories={
        "North-East": ["Eastern Europe", "Scandinavia", "Northern Europe"],
        "South-West": ["Western Europe", "Southern Europe"],
        "Islands": ["Great Britain", "Iceland"]
    },
    continent_troop_gains={
        "North-East": 3,
        "South-West": 2,
        "Islands": 4
    },
    territory_connections={
        "Eastern Europe": ["Northern Europe", "Scandinavia", "Southern Europe"],
        "Scandinavia": ["Iceland", "Eastern Europe", "Northern Europe", "Great Britain"],
        "Northern Europe": ["Scandinavia", "Eastern Europe", "Southern Europe", "Western Europe"],
        "Western Europe": ["Great Britain", "Northern Europe"],
        "Iceland": ["Great Britain", "Scandinavia"]
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
