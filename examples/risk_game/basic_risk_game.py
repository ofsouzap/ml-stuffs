from scenarios.risk_game.world import World
from scenarios.risk_game.runner import Runner
from scenarios.risk_game.player_controller import RandomizedComputerPlayerController, UncheckedConsolePlayerController, AggressiveRandomComputerPlayerContoller
from _debug_timer import CumulativeTimer


class PrintLogger(Runner.LoggerBase):
    def log(self, message: str):
        print(f"INFO: {message}")


class BlankLogger(Runner.LoggerBase):
    def log(self, message: str):
        pass


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
    AggressiveRandomComputerPlayerContoller(),
    AggressiveRandomComputerPlayerContoller(),
]

logger = BlankLogger()

runner = Runner(world, player_controllers, logger, initial_placement_round_count=5)

for _ in range(10000):
    runner.run()

logger.log(f"Winner is player {runner.winner} (end round {runner.get_round_number()+1})")

times = CumulativeTimer.get_all_entries()

for id in sorted(times.keys(), key=lambda x: times[x].time_per_run, reverse=True):
    print(f"{(times[id].time_per_run)/1e-6:.3f}µs/run\t{id}")
