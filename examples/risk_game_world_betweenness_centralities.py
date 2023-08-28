from scenarios.risk_game import FULL_GAME_WORLD
import matplotlib.pyplot as plt
import numpy as np

world = FULL_GAME_WORLD
graph = world.graph

CONTINENT_COLORS = {
    "North America": (0.9, 0.9, 0.4),
    "South America": (0.9, 0.6, 0.3),
    "Africa": (0.9, 0.3, 0.4),
    "Europe": (0.5, 0.7, 0.8),
    "Asia": (0.5, 0.8, 0.4),
    "Australia": (0.6, 0.0, 0.5),
}

territories_list = list(world.iterate_territories())

territory_names = np.array([world.get_territory_name(t) for t in territories_list], dtype=np.str_)
betweenness_centralities = np.array([graph.get_betweenness_centrality(t) for t in territories_list], dtype=np.float64)
territory_continents_list = [world.get_territory_continent(t) for t in territories_list]
territory_colors = np.array([CONTINENT_COLORS[world.get_continent_name(c)] for c in territory_continents_list], dtype=np.float16)

sort_order = np.argsort(betweenness_centralities)
labels = territory_names[sort_order]
values = betweenness_centralities[sort_order]
colors = territory_colors[sort_order]

fig, ax = plt.subplots(1,1)

bar_container = ax.bar(
    x=labels,
    height=values,
    color=colors
)

ax.set_ylabel("Betweenness Centrality")
ax.set_xlabel("Territory")

for label in ax.get_xticklabels():
    label.set_rotation(90)
    label.set_ha("right")
    label.set_fontsize(8)

plt.tight_layout()
plt.show()
