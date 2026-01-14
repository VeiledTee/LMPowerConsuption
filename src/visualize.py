import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd
import matplotlib.pyplot as plt
import re
import math
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullFormatter


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
    size = str(row["params_B"])
    if size.endswith(".0"):
        size = size[:-2]

    # Map context types using existing type mapping
    context_map = {"Base": "Base", "GS": "GS", "FP": "FP"}
    context = context_map.get(row["type"], "Other")

    # Map thinking mode
    think_mode = "Think" if row["thinking"] else "NoThink"

    return f"{size}-{context}-{think_mode}"


def main(csv_path: str):
    # Load Data
    try:
        script_dir = Path(__file__).parent  # Directory where visualize.py is located
        project_root = script_dir.parent  # Go up one level to project root
        results_dir = project_root / "results"
        df = pd.read_csv(results_dir / csv_path)
    except FileNotFoundError:
        print(
            f"Error: '{csv_path}.csv' not found. Please ensure the file is in the same directory."
        )
        return

    # 2. Preprocessing
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

    generate_final_pareto_plots()
    csv_files = ["deepseek_summary.csv", "gemma3_summary.csv", "qwen3_summary.csv"]
    for csv_file in csv_files:
        csv_path = results_dir / csv_file
        if csv_path.exists():
            print(f"\nGenerating scatter plot for: {csv_file}")
            generate_scatter_plots(str(csv_path))

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
    plt.savefig("efficiency_comparison_scale.png", bbox_inches="tight")
    print("Generated efficiency_comparison_scale.png")

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
    plt.savefig("performance_energy_tradeoff_scale.png", bbox_inches="tight")
    print("Generated performance_energy_tradeoff_scale.png")

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

    plt.legend(handles=legend_elements, loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.savefig("pareto_frontier_scale.png", format="svg", dpi=300, bbox_inches="tight")
    print("Generated pareto_frontier_scale.png")

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


def generate_final_pareto_plots():
    # 1. Define Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / "results"

    csv_files = ["deepseek_summary.csv", "gemma3_summary.csv", "qwen3_summary.csv"]
    dfs = []
    for csv_file in csv_files:
        file_path = results_dir / csv_file
        if file_path.exists():
            try:
                dfs.append(pd.read_csv(file_path))
            except Exception:
                pass

    if not dfs:
        return
    df = pd.concat(dfs, ignore_index=True)

    def extract_size(model_name):
        match = re.search(r"(\d+(\.\d+)?)B", str(model_name), re.IGNORECASE)
        return float(match.group(1)) if match else 0.0

    df["model_size"] = df["model"].apply(extract_size)

    target_sizes = [0.6, 1.0, 1.5, 1.7, 4.0, 7.0, 8.0, 12.0, 14.0, 27.0, 32.0]
    min_area, max_area = 50, 1200

    size_map = {
        s: min_area + (i / (len(target_sizes) - 1)) * (max_area - min_area)
        for i, s in enumerate(target_sizes)
    }

    def get_marker_area(val):
        closest = min(target_sizes, key=lambda x: abs(x - val))
        return size_map[closest]

    df["marker_area"] = df["model_size"].apply(get_marker_area)

    # --- 2. METRIC LOGIC ---
    target_col = "total_energy_kg"

    # Fallback safety in case the column name was a typo
    if target_col not in df.columns and "total_emissions_kg" in df.columns:
        print(f"Warning: '{target_col}' not found. Using 'total_emissions_kg' instead.")
        target_col = "total_emissions_kg"

    df["g_CO2"] = df[target_col] * 1000
    df["short_model"] = df["model"].apply(lambda x: x.split(" ")[1].strip())

    # Use ground-truth thinking column
    if "thinking" in df.columns:
        df["is_thinking"] = df["thinking"].astype(bool)
    else:
        # safe default if column missing
        df["is_thinking"] = False

    # Marker Assignment
    def assign_marker(row):
        if row["dataset"] == "HotpotQA":
            return "o" if not row["context_used"] else "s"
        else:
            mapping = {
                "Question Only": "o",
                "GS Paragraph": "s",
                "First Paragraph": "^",
            }
            return mapping.get(row["dataset_version"], "o")

    df["marker"] = df.apply(assign_marker, axis=1)

    # Pareto Logic
    def get_pareto_frontier(ds_df, x_col, y_col):
        points = ds_df[[x_col, y_col]].values
        pareto_mask = np.ones(len(points), dtype=bool)
        for i, point in enumerate(points):
            for j, other_point in enumerate(points):
                if i == j:
                    continue
                if other_point[0] >= point[0] and other_point[1] <= point[1]:
                    if other_point[0] > point[0] or other_point[1] < point[1]:
                        pareto_mask[i] = False
                        break
        return ds_df[pareto_mask].sort_values(by=x_col)

    model_colors = {"DeepSeek": "#1f77b4", "Gemma3": "#2ca02c", "Qwen3": "#ff7f0e"}

    for dataset in ["HotpotQA", "NQ"]:
        ds_df = df[df["dataset"] == dataset]
        if ds_df.empty:
            continue

        pareto_df = get_pareto_frontier(ds_df, "f1", "g_CO2")

        plt.figure(figsize=(12, 6))
        ax = plt.gca()

        # Plot Background by family, marker, and thinking flag
        for fam, color in model_colors.items():
            fam_df = ds_df[ds_df["model"].str.contains(fam, case=False)]
            if fam_df.empty:
                continue

            for marker in fam_df["marker"].unique():
                sub = fam_df[fam_df["marker"] == marker]
                if sub.empty:
                    continue

                # Standard (non-thinking)
                standard = sub[~sub["is_thinking"]]
                if not standard.empty:
                    ax.scatter(
                        standard["f1"],
                        standard["g_CO2"],
                        color=color,
                        marker=marker,
                        s=standard["marker_area"],
                        alpha=0.7,
                        edgecolors="none",
                        zorder=2,
                    )

                # Thinking
                thinking = sub[sub["is_thinking"]]
                if not thinking.empty:
                    # scatter supports hatch on some matplotlib versions; include edgecolors
                    ax.scatter(
                        thinking["f1"],
                        thinking["g_CO2"],
                        facecolor=color,
                        marker=marker,
                        s=thinking["marker_area"],
                        alpha=0.7,
                        edgecolors="black",
                        linewidths=0.6,
                        hatch="////",
                        zorder=3,
                    )

        # Plot Frontier Line
        ax.plot(
            pareto_df["f1"],
            pareto_df["g_CO2"],
            color="black",
            linestyle=":",
            lw=2,
            zorder=4,
        )

        pareto_rows = list(pareto_df.iterrows())

        for i, (idx, row) in enumerate(pareto_rows):
            color = next(
                (
                    c
                    for f, c in model_colors.items()
                    if f.lower() in row["model"].lower()
                ),
                "gray",
            )

            ax.scatter(
                row["f1"],
                row["g_CO2"],
                color=color,
                marker=row["marker"],
                s=row["marker_area"],
                edgecolor="black",
                linewidth=1,
                zorder=5,
            )

            base_height = 25
            horizontal_shift = 45
            if dataset == "NQ":
                if i == 3 or i == 7:
                    xytext_offset = (-horizontal_shift, base_height - 10)
                    connection_style = "arc3,rad=0.1"
                elif i == 5:
                    xytext_offset = (-horizontal_shift, base_height + 10)
                    connection_style = "arc3,rad=0.1"
                else:
                    xytext_offset = (-horizontal_shift, base_height)
                    connection_style = "arc3,rad=0.1"
            else:
                xytext_offset = (-horizontal_shift, base_height)
                connection_style = "arc3,rad=0.1"

            ax.annotate(
                row["short_model"],
                xy=(row["f1"], row["g_CO2"]),
                xytext=xytext_offset,
                textcoords="offset points",
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.2", fc="white", alpha=0.9, ec=color, lw=1.5
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    lw=1,
                    connectionstyle=connection_style,
                    shrinkB=8,
                    shrinkA=2,
                ),
                zorder=10,
            )

        # Y-Axis Formatting
        ax.set_yscale("log")

        # Apply the ticks and the formatter
        y_max_val = df["g_CO2"].max()
        y_upper_limit = math.ceil(y_max_val / 1000) * 1000
        custom_ticks = [100, 500, 1000, 2000, y_upper_limit]
        ax.set_yticks(custom_ticks)
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

        ax.grid(True, which="major", linestyle="--", alpha=0.7)

        ax.set_title(
            f"{dataset if dataset != 'NQ' else 'Natural Questions'} Pareto Frontier",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("F1 Score", fontsize=12)
        ax.set_ylabel("Model Emissions (g of CO$_2$)", fontsize=12)  # Updated Label

        # Legend
        legend_elements = [
            Line2D(
                [0], [0], color="black", linestyle=":", lw=2, label="Pareto Frontier"
            )
        ]
        # Family color patches
        for k, v in model_colors.items():
            legend_elements.append(
                mpatches.Patch(facecolor=v, edgecolor="black", label=k)
            )

        # Context markers (empty face so they match background markers)
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Base",
                markerfacecolor="none",
                markeredgecolor="black",
            )
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="GS",
                markerfacecolor="none",
                markeredgecolor="black",
            )
        )
        if dataset == "NQ":
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    label="FP",
                    markerfacecolor="none",
                    markeredgecolor="black",
                )
            )

        # Conditional Standard / Thinking legend entries (based on dataset)
        has_thinking = ds_df["is_thinking"].any()
        has_standard = (~ds_df["is_thinking"]).any()

        if has_standard:
            legend_elements.append(
                mpatches.Patch(
                    facecolor="gray", edgecolor="black", label="Non-Thinking"
                )
            )

        if has_thinking:
            legend_elements.append(
                mpatches.Patch(
                    facecolor="gray", edgecolor="black", hatch="////", label="Thinking"
                )
            )

        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
        present_sizes = ds_df["model_size"].dropna().unique().tolist()

        # Snap to canonical target sizes
        present_sizes = sorted(
            {min(target_sizes, key=lambda x: abs(x - s)) for s in present_sizes}
        )

        size_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markeredgecolor="black",
                alpha=0.6,
                markersize=math.sqrt(size_map[s]) / 1.5,
                label=f"{s:g}B",
            )
            for s in present_sizes
        ]

        fig = plt.gcf()
        fig.legend(
            handles=size_handles,
            title="Model Size",
            loc="center right",
            bbox_to_anchor=(1.08, 0.5),
            frameon=False,
            labelspacing=1.3,
            title_fontsize=11,
        )
        plt.tight_layout()

        save_path = results_dir / f"pareto_manual_{dataset.lower()}_scale.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved: {save_path}")


def generate_scatter_plots(csv_path):
    """
    Generates scatter plots for HotpotQA and NQ with synced y-axes.
    - Uses SIZE for model scale (ordinal mapping for consistent spacing).
    - Uses static COLOR for model family.
    - Uses HATCHING (////) for thinking models.
    """
    # 1. Load Data
    df = pd.read_csv(csv_path)
    df["g_CO2"] = df["total_energy_kg"] * 1000

    # 2. Extract Size & Family Settings
    def extract_size(model_name):
        match = re.search(r"(\d+(\.\d+)?)B", model_name, re.IGNORECASE)
        return float(match.group(1)) if match else 0

    df["model_size"] = df["model"].apply(extract_size)

    # Identify "Thinking" models (heuristic: checks for 'r1' or 'think' in name)
    df["is_thinking"] = df["thinking"].astype(bool)

    # 3. Define Visual Style based on Family
    filename = Path(csv_path).name.lower()
    if "deepseek" in filename:
        family_color = "#1f77b4"  # Blue
        family_name = "DeepSeek-r1"
    elif "gemma3" in filename:
        family_color = "#2ca02c"  # Green
        family_name = "Gemma 3"
    elif "qwen3" in filename:
        family_color = "#ff7f0e"  # Orange
        family_name = "Qwen3"
    else:
        family_color = "#7f7f7f"  # Gray default
        family_name = "Model"

    # 4. Create Ordinal Size Map
    target_sizes = [0.6, 1.0, 1.5, 1.7, 4.0, 7.0, 8.0, 12.0, 14.0, 27.0, 32.0]

    # Map these sizes to visual marker areas (s) linearly from min to max
    min_area, max_area = 50, 1200
    size_map = {}

    # Normalize ranks 0..N to area range
    for i, size_val in enumerate(target_sizes):
        visual_area = min_area + (i / (len(target_sizes) - 1)) * (max_area - min_area)
        size_map[size_val] = visual_area

    # Fallback for sizes not in list: closest match
    def get_marker_size(val):
        closest = min(target_sizes, key=lambda x: abs(x - val))
        return size_map.get(closest, min_area)

    df["marker_area"] = df["model_size"].apply(get_marker_size)

    # Global Axis Limits
    y_max_val = df["g_CO2"].max()
    y_upper_limit = math.ceil(y_max_val / 1000) * 1000
    custom_ticks = [100, 500, 1000, 2000, y_upper_limit]

    # 5. Setup Plot
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True, constrained_layout=True
    )

    # Main Title
    fig.suptitle(
        f"{family_name} F1 Score vs Total Emissions",
        fontsize=18,
        fontweight="bold",
        y=1.05,
    )

    # Common Axis Settings
    for ax in axes:
        ax.set_yscale("log")
        ax.set_yticks(custom_ticks)
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(df["g_CO2"].min() * 0.8, y_upper_limit * 1.5)
        ax.grid(True, which="major", linestyle="--", alpha=0.6)
        ax.grid(False, which="minor")

    # Helper to plot subsets
    def plot_subset(ax, subset, marker_shape):
        if subset.empty:
            return

        # Split into Thinking and Regular
        thinking = subset[subset["is_thinking"] == True]
        regular = subset[subset["is_thinking"] == False]

        # 1. Plot Regular (Solid Color)
        if not regular.empty:
            ax.scatter(
                regular["f1"],
                regular["g_CO2"],
                s=regular["marker_area"],
                c=family_color,
                marker=marker_shape,
                alpha=0.7,
                edgecolor="black",
                linewidth=1.0,
                label="_nolegend_",
            )

        # 2. Plot Thinking
        if not thinking.empty:
            ax.scatter(
                thinking["f1"],
                thinking["g_CO2"],
                s=thinking["marker_area"],
                c=family_color,
                marker=marker_shape,
                alpha=0.7,
                edgecolor="black",
                linewidth=1.0,
                hatch="////",
                label="_nolegend_",
            )

    # --- Plot 1: HotpotQA ---
    ax1 = axes[0]
    hp = df[df["dataset"] == "HotpotQA"]

    plot_subset(ax1, hp[hp["context_used"] == False], "o")
    plot_subset(ax1, hp[hp["context_used"] == True], "s")

    ax1.set_title("HotpotQA", fontsize=14, fontweight="bold")
    ax1.set_xlabel("F1 Score", fontsize=12)
    ax1.set_ylabel("Model Emissions (g of CO$_2$)", fontsize=12)

    # --- Plot 2: NQ ---
    ax2 = axes[1]
    nq = df[df["dataset"] == "NQ"]

    nq_shapes = {
        "Question Only": "o",  # Base
        "GS Paragraph": "s",  # GS
        "First Paragraph": "^",  # FP
    }

    for version, marker in nq_shapes.items():
        sub = nq[nq["dataset_version"] == version]
        plot_subset(ax2, sub, marker)

    ax2.set_title("Natural Questions", fontsize=14, fontweight="bold")
    ax2.set_xlabel("F1 Score", fontsize=12)

    # --- Legends ---
    # 1. Size Legend - Showing size scaling
    present_sizes = df["model_size"].dropna().unique().tolist()

    # Snap to your canonical target_sizes (same logic as markers)
    present_sizes = sorted(
        {min(target_sizes, key=lambda x: abs(x - s)) for s in present_sizes}
    )

    # Optional: cap legend entries to avoid clutter
    MAX_LEGEND_SIZES = 5
    if len(present_sizes) > MAX_LEGEND_SIZES:
        # evenly sample across range
        idxs = np.linspace(0, len(present_sizes) - 1, MAX_LEGEND_SIZES, dtype=int)
        present_sizes = [present_sizes[i] for i in idxs]

    size_handles = []
    for s in present_sizes:
        size_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markeredgecolor="black",
                alpha=0.6,
                markersize=math.sqrt(size_map[s]) / 1.5,
                label=f"{s:g}B",
            )
        )

    fig.legend(
        handles=size_handles,
        title="Model Size",
        loc="center right",
        bbox_to_anchor=(1.08, 0.5),
        frameon=False,
        labelspacing=1.2,
        title_fontsize=12,
    )

    # 2. Context & Type Legend
    legend_elements = [
        # Context Shapes
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="black",
            label="Base",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="black",
            label="GS",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="black",
            label="FP",
        ),
    ]

    # Conditionally add thinking / standard
    has_thinking = df["is_thinking"].any()
    has_standard = (~df["is_thinking"]).any()

    if has_standard:
        legend_elements.append(
            mpatches.Patch(
                facecolor=family_color, edgecolor="black", label="Non-Thinking"
            )
        )

    if has_thinking:
        legend_elements.append(
            mpatches.Patch(
                facecolor=family_color,
                edgecolor="black",
                hatch="////",
                label="Thinking",
            )
        )

    ax1.legend(
        handles=legend_elements,
        loc="upper left",
        title="Legend",
        framealpha=0.9,
    )
    ax2.legend(
        handles=legend_elements,
        loc="upper left",
        title="Legend",
        framealpha=0.9,
    )

    # Save
    output_dir = Path(csv_path).parent
    save_path = output_dir / f"{family_name}_dataset_breakdown_scale.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    generate_final_pareto_plots()
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / "results"
    csv_files = ["deepseek_summary.csv", "gemma3_summary.csv", "qwen3_summary.csv"]
    for csv_file in csv_files:
        csv_path = results_dir / csv_file
        if csv_path.exists():
            print(f"\nGenerating scatter plot for: {csv_file}")
            generate_scatter_plots(str(csv_path))
