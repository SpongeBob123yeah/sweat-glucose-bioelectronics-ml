import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
DATA = BASE / "results"

df = pd.read_csv(DATA / "association_matrix.csv", index_col=0)
corr_matrix = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
categories = corr_matrix.columns.tolist()
N = len(categories)

colors = ["#6090c1", "#acd2e5", "#f9f0b7", "#f9bc6b", "#e26f49", "#d7312d"]
cmap = LinearSegmentedColormap.from_list("custom_seismic", colors, N=256)
norm = plt.Normalize(vmin=0, vmax=1)

fig = plt.figure(figsize=(14, 14))
gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 0.08])
ax = fig.add_subplot(gs[0], polar=True)

start_angle = np.deg2rad(22.4)
end_angle = 2 * np.pi - start_angle
total_angle = end_angle - start_angle
angles = np.linspace(start_angle, end_angle, N, endpoint=False)

inner_radius = 65.0
ring_width = 0.15
gap = 0.01

for idx, (label, values) in enumerate(corr_matrix.iterrows()):
    bar_colors = cmap(norm(values.values))
    for i, angle in enumerate(angles):
        bar_bottom = inner_radius + idx * (ring_width + gap)
        ax.bar(
            angle,
            height=ring_width,
            width=total_angle / N,
            bottom=bar_bottom,
            color=bar_colors[i],
            edgecolor="none",
            align="edge",
        )

ax.set_theta_offset(np.pi / 2 + start_angle / 2)
ax.set_theta_direction(-1)
ax.set_rlim(inner_radius - 1, inner_radius + N * (ring_width + gap))
ax.spines["polar"].set_visible(False)
ax.grid(False)

ax.set_xticks([])
ax.set_yticks([])

cax_right = fig.add_subplot(gs[1])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax_right, orientation="vertical")
cbar.set_label("Association strength (0–1)", fontsize=11)
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
fig.savefig(FIG / "circular_association_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)
