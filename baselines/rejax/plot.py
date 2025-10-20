import os
import json
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

folder_path = os.path.join(os.path.dirname(__file__), "results")
json_files = [file for file in os.listdir(folder_path) if file.endswith(".json")]

files_by_env = {}
for file in json_files:
    parts = file.split("_")
    # key is env-id
    env_id = "_".join(parts[:2])[6:]
    if env_id not in files_by_env:
        files_by_env[env_id] = []
    # check if file contains 'return' key
    with open(os.path.join(folder_path, file), "r") as f:
        data = json.load(f)
        if "return" not in data:
            continue
    files_by_env[env_id].append(file)

fig, ax = plt.subplots(4, 3, figsize=(11, 9), dpi=150)
for i, env in enumerate(files_by_env):
    files = files_by_env[env]
    if i >= 12:
        break
    algos = {}
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            algo_id = data["algorithm"].upper()
            if "return" not in data:
                continue
            algos[algo_id] = {
                "return": data["return"],
                "step": data["time/step"],
            }

    xs = {k: jnp.asarray(v["step"]) for k, v in algos.items()}  # (T, B)
    returns = {k: jnp.asarray(v["return"]) for k, v in algos.items()}  # (T, B)
    returns_avg = jax.tree.map(lambda x: x.mean(axis=1), returns)  # (T,)
    returns_5 = jax.tree.map(lambda x: jnp.percentile(x, 5, axis=1), returns)
    returns_95 = jax.tree.map(lambda x: jnp.percentile(x, 95, axis=1), returns)

    r = i // 3
    c = i % 3
    colours = [
        "#0077BE",  # Deep Blue
        "#009ECE",  # Sky Blue
        "#DEB887",  # Burlywood
        "#F4A460",  # Sandy Brown
        "#FF7F50",  # Coral
        "#00C3E0",  # Turquoise
        "#00E6F2",  # Light Blue
        "#FFD700",  # Gold
        "#FFA500",  # Orange
        "#FF8C00",  # Dark Orange
    ]
    markers = ["o", "s", "v", "^", "D", "x", "p", "*", "h", "H", "+"]
    for j, algo_id in enumerate(algos):
        ax[r, c].plot(
            xs[algo_id],
            returns_avg[algo_id],
            label=algo_id if i == 0 else "",
            color=colours[j],
            marker=markers[j],
            markersize=3,
        )
        ax[r, c].fill_between(
            xs[algo_id],
            returns_5[algo_id],
            returns_95[algo_id],
            alpha=0.2,
            color=colours[j],
        )

    ax[r, c].grid(axis="y", linestyle=(0, (6, 8)), alpha=0.6)
    # num steps only on last row
    if r == len(files_by_env) // 3 - 1:
        ax[r, c].set_xlabel("Number of steps", fontsize=12)
    # return label on on first column
    if c == 0:
        ax[r, c].set_ylabel("Return", fontsize=12)
    ax[r, c].set_title(env[6:], fontsize=14)
    ax[r, c].tick_params(axis="both", which="major", labelsize=10)
    ax[r, c].set_xlim(0, 1e6)
    ax[r, c].set_ylim(0.0, 1.0)

# legend
legend = fig.legend(
    loc="lower center",
    ncol=fig.axes[0].lines.__len__(),
    bbox_to_anchor=(0.53, -0.035),
    shadow=False,
    frameon=False,
)
fig.tight_layout()
fig.savefig(
    os.path.join(os.path.dirname(__file__), "baselines.png"),
    bbox_extra_artists=(legend,),
    bbox_inches="tight",
)

# Collect data for Markdown table
# markdown_rows = []

# for env in files_by_env:
#     files = files_by_env[env]

#     algos = {}
#     for file in files:
#         file_path = os.path.join(folder_path, file)
#         with open(file_path, "r") as f:
#             data = json.load(f)
#             algo_id = data["algorithm"].upper()
#             if "return" not in data:
#                 continue
#             algos[algo_id] = {
#                 "return": jnp.asarray(data["return"]),  # shape (T, B)
#             }

#     for algo_id, v in algos.items():
#         final_returns = v["return"][-1]  # (B,)
#         avg = float(jnp.mean(final_returns))
#         std = float(jnp.std(final_returns))
#         markdown_rows.append(
#             {
#                 "Environment": env[6:],  # strip 'Navix-'
#                 "Algorithm": algo_id,
#                 "Final Return": f"{avg:.3f} ± {std:.3f}",
#             }
#         )

# # Convert to markdown table
# df = pd.DataFrame(markdown_rows)
# pivot_df = df.pivot(index="Algorithm", columns="Environment", values="Final Return")

# print("\n### 📊 Final Return at Last Training Step (± std)\n")
# table = pivot_df.to_markdown()
# print(table)
# with open(os.path.join(os.path.dirname(__file__), "baselines.md"), "w") as f:
#     f.write(table)

# --- After plot generation ---

# --- After plot generation ---

from collections import defaultdict
import pandas as pd

returns_per_algo = defaultdict(dict)  # algo → env → array (T, B)
steps_per_env = {}

for env in files_by_env:
    files = files_by_env[env]
    for file in files:
        file_path = os.path.join(folder_path, file)
        with open(file_path, "r") as f:
            data = json.load(f)
            algo_id = data["algorithm"].upper()
            returns = jnp.asarray(data["return"])  # (T, B)
            steps = jnp.asarray(data["time/step"])  # (T, B) or (T,)

            returns_per_algo[algo_id][env[6:]] = returns
            steps_per_env[env[6:]] = steps[:, 0] if steps.ndim == 2 else steps

# Create one markdown document
markdown_lines = ["# 📊 Return over Time per Algorithm (Envs as rows)\n"]

for algo, env_returns in returns_per_algo.items():
    markdown_lines.append(f"\n## {algo}\n")
    
    # Collect one step array for columns
    step_arr = list(steps_per_env.values())[0]
    col_labels = [int(s) for s in step_arr]

    row_dicts = {}
    for env, returns in env_returns.items():
        row = {}
        for i, step in enumerate(col_labels):
            mean = float(jnp.mean(returns[i]))
            std = float(jnp.std(returns[i]))
            row[step] = f"{mean:.3f} ± {std:.3f}"
        row_dicts[env] = row

    df = pd.DataFrame.from_dict(row_dicts, orient="index")
    df.index.name = "Environment"
    df.columns.name = "Step"
    markdown_lines.append(df.reset_index().to_markdown(index=False))

# Save markdown document
output_md_path = os.path.join(os.path.dirname(__file__), "baselines.md")
with open(output_md_path, "w") as f:
    f.write("\n".join(markdown_lines))

print(f"Saved markdown table to: {output_md_path}")
