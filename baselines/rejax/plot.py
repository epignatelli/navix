import os
import json
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

folder_path = "/home/uceeepi/repos/navix/baselines/results/rejax"
json_files = [file for file in os.listdir(folder_path) if file.endswith(".json")]

files_by_env = {}
for file in json_files:
    parts = file.split("_")
    # key is env-id
    key = "_".join(parts[:2])[6:]
    if key not in files_by_env:
        files_by_env[key] = []
    files_by_env[key].append(file)

fig, ax = plt.subplots(4, 3, figsize=(11, 9), dpi=150)
i = 0
for env in files_by_env:
    files = files_by_env[env]

    algos = {}
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            key = data["algorithm"].upper()
            if key.startswith("D"):
                key = "DDQN"
            algos[key] = {
                "return": data["return"],
                "step": data["step"],
            }

    xs = {k: jnp.asarray(v["step"]) for k, v in algos.items()}  # (T, B)
    returns = {k: jnp.asarray(v["return"]) for k, v in algos.items()}  # (T, B)
    returns_avg = jax.tree.map(lambda x: x.mean(axis=1), returns)  # (T,)
    returns_5 = jax.tree.map(lambda x: jnp.percentile(x, 5, axis=1), returns)
    returns_95 = jax.tree.map(lambda x: jnp.percentile(x, 95, axis=1), returns)

    r, c = i // 3, i % 3
    colours = [
        "#0077BE",  # Deep Blue
        "#009ECE",  # Sky Blue
        # "#00C3E0",  # Turquoise
        # "#00E6F2",  # Light Blue
        # "#FFD700",  # Gold
        # "#FFA500",  # Orange
        # "#FF8C00",  # Dark Orange
        # "#F4A460",  # Sandy Brown
        "#DEB887",  # Burlywood
        # "#FF7F50",  # Coral
    ]
    markers = ["o", "s", "v"]
    for j, algo in enumerate(algos):
        ax[r, c].plot(
            xs[algo],
            returns_avg[algo],
            label=algo if i == 0 else "",
            color=colours[j],
            marker=markers[j],
            markersize=3,
        )
        ax[r, c].fill_between(
            xs[algo],
            returns_5[algo],
            returns_95[algo],
            alpha=0.2,
            color=colours[j],
        )
    i += 1

    ax[r, c].grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)
    ax[r, c].set_xlabel("Number of steps", fontsize=12)
    ax[r, c].set_ylabel("Return", fontsize=12)
    ax[r, c].set_title(env[6:], fontsize=14)
    ax[r, c].tick_params(axis="both", which="major", labelsize=10)
legend = fig.legend(
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.53, -0.05),
    shadow=False,
    frameon=False,
)
fig.tight_layout()
fig.savefig(
    os.path.join(os.path.dirname(__file__), "baselines.png"),
    bbox_extra_artists=(legend,),
    bbox_inches="tight",
)
