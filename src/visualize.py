import seaborn as sns
import re
from matplotlib.colors import LogNorm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


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


def generate_final_pareto_plots():
    # 1. Define Paths - Use absolute path
    # The script is in project/src/, results are in project/results/
    script_dir = Path(__file__).parent  # Directory where visualize.py is located
    project_root = script_dir.parent  # Go up one level to project root
    results_dir = project_root / "results"

    print(f"Looking for CSV files in: {results_dir}")

    # 2. Load and Combine Data with better error handling
    csv_files = ["deepseek_summary.csv", "gemma3_summary.csv", "qwen3_summary.csv"]
    dfs = []

    # Check each file individually
    for csv_file in csv_files:
        file_path = results_dir / csv_file
        if file_path.exists():
            try:
                df_temp = pd.read_csv(file_path)
                dfs.append(df_temp)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
        else:
            print(f"Not found: {csv_file}")

    if not dfs:
        print("Error: No CSV files were successfully loaded.")
        # List what's actually in the directory
        for f in results_dir.iterdir():
            if f.is_file():
                print(f"  - {f.name}")
        return

    # Combine dataframes
    df = pd.concat(dfs, ignore_index=True)

    # 3. Preprocessing
    df["g_CO2"] = df["emissions_kg_per_question"] * 1000
    df["short_model"] = df["model"].apply(lambda x: x.split("(")[0].strip())

    # 4. Marker Assignment Logic
    def assign_marker(row):
        if row["dataset"] == "HotpotQA":
            # Hotpot: Base (False) vs Gold Standard (True)
            return "o" if not row["context_used"] else "s"
        else:
            # NQ: Map based on dataset_version
            mapping = {
                "Question Only": "o",
                "GS Paragraph": "s",
                "First Paragraph": "^",
            }
            return mapping.get(row["dataset_version"], "o")

    df["marker"] = df.apply(assign_marker, axis=1)

    # 5. Pareto Frontier Logic
    def get_pareto_frontier(ds_df, x_col, y_col):
        """Finds non-dominated points (Maximize X, Minimize Y)"""
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

    # 6. Plotting
    model_colors = {"DeepSeek": "#1f77b4", "Gemma3": "#2ca02c", "Qwen3": "#ff7f0e"}

    for dataset in ["HotpotQA", "NQ"]:
        ds_df = df[df["dataset"] == dataset]
        if ds_df.empty:
            print(f"No data found for dataset: {dataset}")
            continue

        print(f"\nProcessing {dataset}: {len(ds_df)} rows")
        pareto_df = get_pareto_frontier(ds_df, "f1", "g_CO2")

        plt.figure(figsize=(12, 5))

        # Plot background (All points with their specific shapes/colors)
        for fam, color in model_colors.items():
            fam_df = ds_df[ds_df["model"].str.contains(fam, case=False)]
            if not fam_df.empty:
                for marker in fam_df["marker"].unique():
                    sub = fam_df[fam_df["marker"] == marker]
                    plt.scatter(
                        sub["f1"],
                        sub["g_CO2"],
                        color=color,
                        marker=marker,
                        s=60,
                        alpha=0.5,
                        edgecolors="none",
                        label=f"{fam}_{marker}",
                    )

        # Draw the Frontier Line (DOTTED BLACK)
        plt.step(
            pareto_df["f1"],
            pareto_df["g_CO2"],
            where="post",
            color="black",
            linestyle=":",
            lw=2.5,
            label="Pareto Frontier",
            zorder=4,
        )

        # Draw Frontier points with HEAVY BLACK OUTLINE and original colors
        for i, (idx, row) in enumerate(pareto_df.iterrows()):
            # Determine color based on model
            color = None
            for fam, fam_color in model_colors.items():
                if fam.lower() in row["model"].lower():
                    color = fam_color
                    break

            plt.scatter(
                row["f1"],
                row["g_CO2"],
                color=color,
                marker=row["marker"],
                s=180,
                edgecolor="black",
                linewidth=3,
                zorder=5,
            )

            # Spaced labels with arrows (use black border for consistency)
            y_off = 45 if i % 2 == 0 else -55
            plt.annotate(
                row["short_model"],
                (row["f1"], row["g_CO2"]),
                xytext=(35, y_off),
                textcoords="offset points",
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="black", lw=1.5
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    alpha=0.8,
                    connectionstyle="arc3,rad=0.1",
                ),
                zorder=6,
            )

        plt.title(
            f"Pareto Frontier: Performance vs Carbon Emissions ({dataset})",
            fontsize=18,
            fontweight="bold",
            pad=25,
        )
        plt.xlabel("F1 Score", fontsize=14)
        plt.ylabel("g of CO$_2$ per question", fontsize=14)

        # UPDATED LEGEND with better visual distinction
        import matplotlib.patches as mpatches

        legend_elements = [
            Line2D(
                [0], [0], color="black", linestyle=":", lw=2.5, label="Pareto Frontier"
            )
        ]

        # Model families - use COLORED SQUARES instead of circles
        for k, v in model_colors.items():
            legend_elements.append(
                mpatches.Patch(color=v, label=k, edgecolor="black", linewidth=1)
            )

        legend_elements.append(Line2D([0], [0], color="w", label=""))  # Spacer

        # Context types - use black outlines on grey shapes
        grey_colour = "#777777"
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Base (No Context)",
                markerfacecolor=grey_colour,
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=1.5,
            )
        )
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="GS Paragraph",
                markerfacecolor=grey_colour,
                markersize=12,
                markeredgecolor="black",
                markeredgewidth=1.5,
            )
        )
        if dataset == "NQ":
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    label="First Paragraph",
                    markerfacecolor=grey_colour,
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                )
            )

        plt.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            shadow=True,
            framealpha=0.95,
        )
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        save_path = results_dir / f"pareto_all_families_{dataset.lower()}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  Plot saved to: {save_path}")


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
    Uses clearer blue-to-red colormap and improved legend design.
    """
    # 1. Load and Clean Data
    df = pd.read_csv(csv_path)
    df["g_CO2"] = df["emissions_kg_per_question"] * 1000

    def extract_size(model_name):
        match = re.search(r"(\d+(\.\d+)?)B", model_name)
        return float(match.group(1)) if match else 0

    df["model_size"] = df["model"].apply(extract_size)
    df = df.sort_values("model_size")

    # Determine global y-axis limit
    y_max = df["g_CO2"].max() * 1.1  # Add 10% headroom

    # 2. Setup Plot with constrained_layout to avoid tight_layout issues
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True, constrained_layout=True
    )
    model_name = Path(csv_path).stem.replace("_summary", "").capitalize()
    fig.suptitle(
        f"{model_name} Model Performance vs Carbon Emissions",
        fontsize=16,
        fontweight="bold",
        y=1.05,
    )

    # 3. Use blue-to-red colormap
    cmap = plt.cm.coolwarm

    # Logarithmic normalization for better contrast between close model sizes
    norm = LogNorm(vmin=df["model_size"].min(), vmax=df["model_size"].max())

    # Define distinct marker properties for better visibility
    marker_size = 60
    edge_width = 1.5

    # --- Plot 1: HotpotQA ---
    ax1 = axes[0]
    hotpot = df[df["dataset"] == "HotpotQA"]

    # Base (Circle)
    base_hp = hotpot[hotpot["context_used"] == False]
    scatter1 = ax1.scatter(
        base_hp["f1"],
        base_hp["g_CO2"],
        c=base_hp["model_size"],
        cmap=cmap,
        norm=norm,
        marker="o",
        s=marker_size,
        alpha=0.8,
        edgecolor="black",
        linewidth=edge_width,
    )

    # GS Paragraph (Square)
    rag_hp = hotpot[hotpot["context_used"] == True]
    scatter2 = ax1.scatter(
        rag_hp["f1"],
        rag_hp["g_CO2"],
        c=rag_hp["model_size"],
        cmap=cmap,
        norm=norm,
        marker="s",
        s=marker_size,
        alpha=0.8,
        edgecolor="black",
        linewidth=edge_width,
    )

    ax1.set_title("HotpotQA Dataset", fontsize=14, fontweight="bold")
    ax1.set_xlabel("F1 Score", fontsize=12)
    ax1.set_ylabel("g of CO$_2$ per question", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_ylim(0, y_max)

    # --- Plot 2: NQ ---
    ax2 = axes[1]
    nq = df[df["dataset"] == "NQ"]
    nq_markers = {
        "Question Only": ("o", "Base (No Context)"),
        "GS Paragraph": ("s", "GS Paragraph"),
        "First Paragraph": ("^", "First Paragraph"),
    }

    for version, (marker, label) in nq_markers.items():
        subset = nq[nq["dataset_version"] == version]
        if not subset.empty:
            ax2.scatter(
                subset["f1"],
                subset["g_CO2"],
                c=subset["model_size"],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=marker_size,
                alpha=0.8,
                edgecolor="black",
                linewidth=edge_width,
            )

    ax2.set_title("NQ Dataset", fontsize=14, fontweight="bold")
    ax2.set_xlabel("F1 Score", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_ylim(0, y_max)

    # --- 4. Colorbar with improved ticks and labels ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Get unique model sizes and create nicely spaced ticks
    unique_sizes = sorted(df["model_size"].unique())

    # Create a colorbar with model sizes clearly labeled
    cbar = fig.colorbar(
        sm, ax=axes, shrink=0.8, location="right", pad=0.02, ticks=unique_sizes
    )
    cbar.set_label(
        "Model Size (Billions of Parameters)", fontsize=11, fontweight="bold"
    )
    cbar.ax.set_yticklabels([f"{s:g}B" for s in unique_sizes], fontsize=10)

    # --- 5. Unified context type legend ---
    # Create a single unified legend for context types
    grey_colour = "#777777"
    legend_marker_size = 10

    # For HotpotQA plot
    hotpot_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Base (No Context)",
            markerfacecolor=grey_colour,
            markersize=legend_marker_size,
            markeredgecolor="black",
            markeredgewidth=1.5,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="GS Paragraph",
            markerfacecolor=grey_colour,
            markersize=legend_marker_size,
            markeredgecolor="black",
            markeredgewidth=1.5,
            linewidth=0,
        ),
    ]

    # For NQ plot (includes triangle for First Paragraph)
    nq_legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Base (No Context)",
            markerfacecolor=grey_colour,
            markersize=legend_marker_size,
            markeredgecolor="black",
            markeredgewidth=1.5,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="GS Paragraph",
            markerfacecolor=grey_colour,
            markersize=legend_marker_size,
            markeredgecolor="black",
            markeredgewidth=1.5,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="First Paragraph",
            markerfacecolor=grey_colour,
            markersize=legend_marker_size,
            markeredgecolor="black",
            markeredgewidth=1.5,
            linewidth=0,
        ),
    ]

    # Add legends to respective plots
    ax1.legend(
        handles=hotpot_legend_elements,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        title="Context Type",
        title_fontsize=11,
        fontsize=10,
    )

    ax2.legend(
        handles=nq_legend_elements,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        title="Context Type",
        title_fontsize=11,
        fontsize=10,
    )

    output_dir = Path(csv_path).parent
    model_name = Path(csv_path).stem.split("_")[0]
    save_path = output_dir / f"{model_name.capitalize()}_dataset_breakdown.png"

    # Use bbox_inches='tight' to ensure the title isn't cut off!
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Plot saved to: {save_path}")


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
    plt.savefig("efficiency_comparison.png", bbox_inches="tight")
    print("Generated efficiency_comparison.png")

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
    plt.savefig("performance_energy_tradeoff.png", bbox_inches="tight")
    print("Generated performance_energy_tradeoff.png")

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
    plt.savefig("pareto_frontier.png", format="svg", dpi=300, bbox_inches="tight")
    print("Generated pareto_frontier.png")

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
    main("qwen3_summary.csv")
