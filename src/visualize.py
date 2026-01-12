from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from matplotlib.colors import LogNorm


def parse_model_size(model_name):
    """Extracts model size in Billions (B) from string."""
    match = re.search(r"(\d+\.?\d*)B", str(model_name))
    if match:
        return float(match.group(1))
    return 0.0


def map_dataset_type(version):
    """Maps dataset version to simplified types for plotting."""
    version = str(version).lower()
    if "question only" in version:
        return "Base"
    elif "gs paragraph" in version:
        return "GS"
    elif "first paragraph" in version:
        return "FP"
    return "Other"


def get_marker(dtype):
    """Returns marker style based on dataset type."""
    if dtype == "Base":
        return "o"  # Circle
    if dtype == "GS":
        return "^"  # Triangle
    if dtype == "FP":
        return "s"  # Square
    return "x"


def find_pareto_frontier(df, x_col, y_col, maximize_y=True):
    """Find Pareto optimal configurations."""
    points = df[[x_col, y_col]].values
    pareto_mask = np.ones(len(points), dtype=bool)

    for i, point in enumerate(points):
        if pareto_mask[i]:
            for j, other_point in enumerate(points):
                if i != j and pareto_mask[j]:
                    if maximize_y:
                        if (
                            other_point[0] <= point[0]
                            and other_point[1] >= point[1]
                            and (other_point[0] < point[0] or other_point[1] > point[1])
                        ):
                            pareto_mask[i] = False
                            break
                    else:
                        if (
                            other_point[0] <= point[0]
                            and other_point[1] <= point[1]
                            and (other_point[0] < point[0] or other_point[1] < point[1])
                        ):
                            pareto_mask[i] = False
                            break
    return pareto_mask


def get_pareto_frontier_detailed(df, x_col, y_col):
    """
    Returns the Pareto frontier points for maximization of y (F1) and minimization of x (Energy)
    """
    points = df[[x_col, y_col, "config_label"]].values
    pareto_points = []

    for point in points:
        is_dominated = False
        for other_point in points:
            # Check if other_point dominates point (lower energy AND higher F1)
            if (
                other_point[0] <= point[0]
                and other_point[1] >= point[1]
                and (other_point[0] < point[0] or other_point[1] > point[1])
            ):
                is_dominated = True
                break
        if not is_dominated:
            pareto_points.append(point)

    # Sort by x-axis for clean plotting
    pareto_points = sorted(pareto_points, key=lambda x: x[0])
    return np.array(pareto_points)


def create_config_label(row):
    """Create standardized configuration labels for annotations"""
    # Extract size (0.6, 1.7, 4, 8, 14, 32)
    size = str(row["params_B"])
    if size.endswith(".0"):
        size = size[:-2]

    # Map context types using existing type mapping
    context_map = {"Base": "Base", "GS": "GS", "FP": "FP"}
    context = context_map.get(row["type"], "Other")

    # Map thinking mode
    think_mode = "Think" if row["thinking"] else "NoThink"

    return f"{size}-{context}-{think_mode}"


def generate_scatter_plots(csv_path):
    """
    Generates scatter plots for HotpotQA and NQ with synced y-axes
    and a logarithmic color scale for distinct model sizes.
    """
    # 1. Load and Clean Data
    df = pd.read_csv(csv_path)
    df['g_CO2'] = df['emissions_kg_per_question'] * 1000

    def extract_size(model_name):
        match = re.search(r"(\d+(\.\d+)?)B", model_name)
        return float(match.group(1)) if match else 0

    df['model_size'] = df['model'].apply(extract_size)
    df = df.sort_values('model_size')

    # Determine global y-axis limit
    y_max = df['g_CO2'].max() * 1.1  # Add 10% headroom

    # 2. Setup Plot (sharey=True ensures axes are synced)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout='constrained')
    fig.suptitle('Qwen3 Model Performance vs Carbon Emissions', fontsize=16, fontweight='bold')

    # 3. Logarithmic Normalization for color contrast
    cmap = plt.cm.plasma
    norm = LogNorm(vmin=df['model_size'].min(), vmax=df['model_size'].max())

    # --- Plot 1: HotpotQA ---
    ax1 = axes[0]
    hotpot = df[df['dataset'] == 'HotpotQA']

    # Base (Circle)
    base_hp = hotpot[hotpot['context_used'] == False]
    ax1.scatter(base_hp['f1'], base_hp['g_CO2'], c=base_hp['model_size'],
                cmap=cmap, norm=norm, marker='o', s=50,
                label='No Context', alpha=0.8, edgecolor='black')

    # GS Paragraph (Square)
    rag_hp = hotpot[hotpot['context_used'] == True]
    ax1.scatter(rag_hp['f1'], rag_hp['g_CO2'], c=rag_hp['model_size'],
                cmap=cmap, norm=norm, marker='s', s=50,
                label='GS Paragraph', alpha=0.8, edgecolor='black')

    ax1.set_title('HotpotQA Dataset', fontsize=14)
    ax1.set_xlabel('F1 Score', fontsize=12)
    ax1.set_ylabel('g of CO$_2$', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend()
    ax1.set_ylim(0, y_max)

    # --- Plot 2: NQ ---
    ax2 = axes[1]
    nq = df[df['dataset'] == 'NQ']
    nq_markers = {
        'Question Only': ('o', 'No Context'),
        'GS Paragraph': ('s', 'GS Paragraph'),
        'First Paragraph': ('^', 'First Paragraph')
    }

    for version, (marker, label) in nq_markers.items():
        subset = nq[nq['dataset_version'] == version]
        if not subset.empty:
            ax2.scatter(subset['f1'], subset['g_CO2'], c=subset['model_size'],
                        cmap=cmap, norm=norm, marker=marker, s=50,
                        label=label, alpha=0.8, edgecolor='black')

    ax2.set_title('NQ Dataset', fontsize=14)
    ax2.set_xlabel('F1 Score', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend()
    ax2.set_ylim(0, y_max)

    # --- 4. Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    unique_sizes = sorted(df['model_size'].unique())
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, location='right', ticks=unique_sizes)
    cbar.set_label('Model Size (B)', fontsize=12)
    cbar.ax.set_yticklabels([f'{s}B' for s in unique_sizes])

    output_dir = Path(csv_path).parent
    model_name = Path(csv_path).stem.split('_')[0]
    save_path = output_dir / f"{model_name.capitalize()}_dataset_breakdown.svg"

    # Use bbox_inches='tight' to ensure the title isn't cut off!
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

def main(csv_path: str):
    # Load Data
    try:
        df = pd.read_csv(
            fr"C:\Users\Ethan\Documents\PhD\LMPowerConsuption\results\{csv_path}"
        )
    except FileNotFoundError:
        print(
            "Error: 'qwen3_summary.csv' not found. Please ensure the file is in the same directory."
        )
        return

    # 2. Preprocessing
    # Extract Model Size
    df["params_B"] = df["model"].apply(parse_model_size)

    # Create ordinal mapping for even spacing on X-axis
    unique_sizes = sorted(df["params_B"].unique())
    size_map = {size: i for i, size in enumerate(unique_sizes)}
    df["size_idx"] = df["params_B"].map(size_map)

    # Normalize Dataset Version Types
    df["type"] = df["dataset_version"].apply(map_dataset_type)

    # Calculate Performance per Energy (F1 / Total kg)
    # Check for zeros in the new total energy column
    if "total_energy_kg" in df.columns:
        df["total_energy_kg"] = df["total_energy_kg"].replace(0, np.nan)
    else:
        print("ERROR: 'total_energy_kg' column not found!")
        return

    df["perf_per_energy"] = df["f1"] / df["total_energy_kg"]

    # The original avg energy check is fine to keep for safety for plots that still use avg energy
    df["emissions_kg_per_question"] = df["emissions_kg_per_question"].replace(0, np.nan)

    # Calculate additional efficiency metrics (these still use avg energy/tokens/time)
    df["performance_per_token"] = df["f1"] / df["pred_tokens_per_question"]
    df["performance_per_second"] = df["f1"] / df["time_s_per_question"]
    df["performance_per_emission"] = df["f1"] / df["emissions_kg_per_question"]
    df["tokens_per_f1"] = df["pred_tokens_per_question"] / df["f1"]

    # Create config labels for annotations
    df["config_label"] = df.apply(create_config_label, axis=1)

    generate_scatter_plots(fr"C:\Users\Ethan\Documents\PhD\LMPowerConsuption\results\{csv_path}")

    # Set general style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    # Performance per Energy (Split by Thinking)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    modes = [False, True]
    mode_names = ["Non-Thinking", "Thinking"]
    type_palette = {"Base": "#2ecc71", "GS": "#9b59b6", "FP": "#f1c40f"}

    for i, mode in enumerate(modes):
        subset = df[df["thinking"] == mode]
        ax = axes[i]

        sns.lineplot(
            data=subset,
            x="size_idx",
            y="perf_per_energy",
            hue="type",
            palette=type_palette,
            ax=ax,
            legend=False,
            markers=False,
            sort=True,
        )

        sns.scatterplot(
            data=subset,
            x="size_idx",
            y="perf_per_energy",
            hue="type",
            style="type",
            markers={"Base": "o", "GS": "^", "FP": "s"},
            s=100,
            palette=type_palette,
            ax=ax,
        )

        ax.set_title(f"{mode_names[i]} Models: Efficiency")
        ax.set_xlabel("Model Parameters (Billions)")
        ax.set_xticks(list(size_map.values()))
        ax.set_xticklabels(list(size_map.keys()))

        if i == 0:

            ax.set_ylabel("Efficiency (F1 / Total kg)")
        else:
            ax.set_ylabel("")

        ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig("efficiency_comparison.svg", bbox_inches="tight")
    print("Generated efficiency_comparison.svg")

    # Performance-Energy Trade-off Scatter Plot
    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        df["f1"],
        df["total_energy_kg"],
        c=df["params_B"],
        s=100,
        alpha=0.7,
        cmap="viridis",
    )
    plt.colorbar(scatter, label="Model Size (B)")
    plt.xlabel("F1 Score")

    plt.ylabel("Total Energy Consumption (kg)")
    plt.title("Performance vs Total Energy Trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("performance_energy_tradeoff.svg", bbox_inches="tight")
    print("Generated performance_energy_tradeoff.svg")

    # Pareto Frontier Analysis

    pareto_points = get_pareto_frontier_detailed(df, "total_energy_kg", "f1")

    plt.figure(figsize=(14, 9))

    # color mapping: Thinking -> blue, Non-thinking -> green
    thinking_color_map = {True: "#E66100", False: "#5D3A9B"}

    # Plot all points individually so we can set both color (thinking) and marker (context)
    for _, row in df.iterrows():
        dtype = row["type"]
        marker = get_marker(dtype)
        color = thinking_color_map[row["thinking"]]

        plt.scatter(
            row["total_energy_kg"],
            row["f1"],
            c=color,
            marker=marker,
            s=80,
            alpha=0.8,
            edgecolors="none",
            zorder=2,
        )

    # Plot dotted black Pareto frontier line
    plt.plot(
        pareto_points[:, 0],
        pareto_points[:, 1],
        color="k",
        linestyle=":",  # dotted
        linewidth=4.0,
        alpha=0.9,
        label="Pareto Frontier",
        zorder=3,
    )

    # Plot Pareto optimal points: same color mapping but with black border
    pareto_colors = [
        (
            thinking_color_map.get(
                bool(df.loc[df["config_label"] == p[2], "thinking"].iloc[0]), "#2ca02c"
            )
            if not df.loc[df["config_label"] == p[2], "thinking"].empty
            else "#2ca02c"
        )
        for p in pareto_points
    ]
    # determine markers for pareto points (respect context)
    pareto_markers = []
    for p in pareto_points:
        row = df[df["config_label"] == p[2]]
        if not row.empty:
            dtype = row.iloc[0]["type"]
        else:
            dtype = "Base"
        pareto_markers.append(get_marker(dtype))

    # scatter each pareto point so we can set its marker individually and keep a black edge
    for (x, y, cfg), color, marker in zip(pareto_points, pareto_colors, pareto_markers):
        plt.scatter(
            x,
            y,
            c=color,
            marker=marker,
            s=150,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

    # Annotations
    key_configs = [
        "1.7-GS-NoThink",
        "1.7-GS-Think",
        "32-Base-NoThink",
    ]

    for point in pareto_points:
        if point[2] in key_configs:
            continue
        if "-Think" in point[2]:
            plt.annotate(
                point[2],
                (point[0], point[1]),
                xytext=(25, -25),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=-0.2",
                    shrinkB=8,  # <-- prevents overlap
                    color="black",
                    lw=1.0,
                ),
                ha="left",
                va="top",
            )
        else:
            plt.annotate(
                point[2],
                (point[0], point[1]),
                xytext=(35, 10),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=0.2",
                    shrinkB=8,
                    color="black",
                    lw=1.0,
                ),
            )

    for config in key_configs:
        point_data = df[df["config_label"] == config]
        if not point_data.empty:
            x_pos = point_data["total_energy_kg"].values[0]
            y_pos = point_data["f1"].values[0]
            plt.annotate(
                config,
                (x_pos, y_pos),
                xytext=(15, -25),
                textcoords="offset points",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="white",
                    alpha=0.95,
                    edgecolor="black",
                    linewidth=1.2,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=-0.2",
                    shrinkB=8,
                    color="red",
                    lw=1.5,
                ),
                ha="left",
                va="top",
            )

    plt.xlabel("Total Energy Consumption (kg)", fontsize=13, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=13, fontweight="bold")
    plt.title(
        "Pareto Frontier: F1 Score vs. Total Energy Consumption\nQwen3 Model Family on Natural Questions",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    plt.xscale("log")
    plt.grid(True, alpha=0.3, linestyle="--")

    # Build legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=thinking_color_map[False],
            markersize=10,
            label="Non-Thinking",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=thinking_color_map[True],
            markersize=10,
            label="Thinking",
        ),
        Line2D(
            [0],
            [0],
            marker=get_marker("Base"),
            color="w",
            markerfacecolor="w",
            markeredgecolor="k",
            markersize=10,
            label="Base",
        ),
        Line2D(
            [0],
            [0],
            marker=get_marker("GS"),
            color="w",
            markerfacecolor="w",
            markeredgecolor="k",
            markersize=10,
            label="GS",
        ),
        Line2D(
            [0],
            [0],
            marker=get_marker("FP"),
            color="w",
            markerfacecolor="w",
            markeredgecolor="k",
            markersize=10,
            label="FP",
        ),
        Line2D(
            [0], [0], color="k", linestyle=":", linewidth=2.0, label="Pareto Frontier"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=10,
            label="Pareto Optimal Border",
        ),
    ]

    plt.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("pareto_frontier.svg", format="svg", dpi=300, bbox_inches="tight")
    print("Generated pareto_frontier.svg")

    # Print the Pareto optimal configurations with details
    print("\n" + "=" * 70)
    print("PARETO OPTIMAL CONFIGURATIONS (SIZE-CONTEXT-THINK)")
    print("=" * 70)
    pareto_df = pd.DataFrame(pareto_points, columns=["energy", "f1", "config"])
    pareto_df = pareto_df.sort_values("energy")
    for i, row in pareto_df.iterrows():
        print(
            f"{row['config']:20} | F1: {row['f1']:.4f} | Total Energy: {row['energy']:.6f} kg | PPE: {row['f1'] / row['energy']:.1f}"
        )

    # Additional analysis
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)
    best_small = df[df["config_label"] == "1.7-GS-NoThink"].iloc[0]
    best_large = df[df["config_label"] == "32-Base-NoThink"].iloc[0]

    # print(f"Best Small (1.7-GS-NoThink):")
    # print(
    #     f"  F1: {best_small['f1']:.4f}, Total Energy: {best_small['total_energy_kg']:.6f} kg, PPE: {best_small['perf_per_energy']:.1f}"
    # )
    # print(f"Best Large (32-Base-NoThink):")
    # print(
    #     f"  F1: {best_large['f1']:.4f}, Total Energy: {best_large['total_energy_kg']:.6f} kg, PPE: {best_large['perf_per_energy']:.1f}"
    # )
    # # The improvement comparison now correctly uses the new 'perf_per_energy' which is F1/Total Energy
    # print(
    #     f"Improvement: F1 +{((best_small['f1'] / best_large['f1']) - 1) * 100:.1f}%, PPE +{((best_small['perf_per_energy'] / best_large['perf_per_energy']) - 1) * 100:.1f}%"
    # )

    # TABLE: Best Small vs Largest
    unique_sizes = sorted(df["params_B"].unique())

    if len(unique_sizes) < 2:
        print("Not enough model sizes to compare small vs large.")
        return

    small_sizes = unique_sizes[:2]
    large_size = unique_sizes[-1]

    print(f"\nComparing Small Models ({small_sizes}B) vs Large Model ({large_size}B)")

    small_df = df[df["params_B"].isin(small_sizes)].copy()
    large_df = df[df["params_B"] == large_size].copy()

    best_small = small_df.loc[small_df.groupby("params_B")["f1"].idxmax()]
    best_large = large_df.sort_values("f1", ascending=False).head(2)

    comparison_table = pd.concat([best_small, best_large])

    cols_to_show = [
        "model",
        "type",
        "thinking",
        "params_B",
        "f1",
        "total_energy_kg",
        "perf_per_energy",
    ]

    final_table = comparison_table[cols_to_show].sort_values("params_B")

    final_table.to_csv("best_config_comparison.csv", index=False)

    print("\n--- Best Configurations Table ---")
    print(final_table.to_string(index=False))

    print("\n=== All plots generated successfully! ===")


if __name__ == "__main__":
    main("deepseek_summary.csv")
