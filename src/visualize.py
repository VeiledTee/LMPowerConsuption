import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np


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
        return "FPar"
    return "Other"


def get_marker(dtype):
    """Returns marker style based on dataset type."""
    if dtype == "Base":
        return "o"  # Circle
    if dtype == "GS":
        return "^"  # Triangle
    if dtype == "FPar":
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
    Returns the Pareto frontier points for maximization of both x and y
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
    context_map = {"Base": "B", "GS": "GS", "FPar": "FP"}
    context = context_map.get(row["type"], "Other")

    # Map thinking mode
    think_mode = "Think" if row["thinking"] else "NoThink"

    return f"{size}-{context}-{think_mode}"


def main():
    # 1. Load Data
    try:
        df = pd.read_csv(
            "/home/penguins/Documents/LMPowerConsumption/results/qwen3_summary.csv"
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

    # Calculate Performance per Energy (F1 / kWh)
    df["energy_kWh_per_question"] = df["energy_kWh_per_question"].replace(0, np.nan)
    df["perf_per_energy"] = df["f1"] / df["energy_kWh_per_question"]

    # Calculate additional efficiency metrics
    df["performance_per_token"] = df["f1"] / df["pred_tokens_per_question"]
    df["performance_per_second"] = df["f1"] / df["time_s_per_question"]
    df["performance_per_emission"] = df["f1"] / df["emissions_kg_per_question"]
    df["tokens_per_f1"] = df["pred_tokens_per_question"] / df["f1"]

    # Create config labels for annotations
    df["config_label"] = df.apply(create_config_label, axis=1)

    # Set general style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    # ---------------------------------------------------------
    # PLOT 1: Performance vs Model Size
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="size_idx",
        y="f1",
        hue="thinking",
        style="thinking",
        markers=True,
        dashes=False,
        palette=["#e74c3c", "#3498db"],
    )
    plt.title("Performance vs Model Size: F1 Score")
    plt.xlabel("Model Parameters (Billions)")
    plt.ylabel("F1 Score")
    plt.xticks(list(size_map.values()), list(size_map.keys()))
    plt.legend(title="Thinking Mode")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("performance_vs_size.svg", bbox_inches="tight")
    print("Generated performance_vs_size.svg")

    # ---------------------------------------------------------
    # PLOT 2: Energy vs Model Size
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="size_idx",
        y="energy_kWh_per_question",
        hue="thinking",
        style="thinking",
        markers=True,
        dashes=False,
        palette=["#e74c3c", "#3498db"],
    )
    plt.title("Energy Consumption vs Model Size")
    plt.xlabel("Model Parameters (Billions)")
    plt.ylabel("Energy (kWh per Question)")
    plt.yscale("log")
    plt.xticks(list(size_map.values()), list(size_map.keys()))
    plt.legend(title="Thinking Mode")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("energy_vs_size.svg", bbox_inches="tight")
    print("Generated energy_vs_size.svg")

    # ---------------------------------------------------------
    # PLOT 3 & 4: Performance per Energy (Split by Thinking)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    modes = [False, True]
    mode_names = ["Non-Thinking", "Thinking"]
    type_palette = {"Base": "#2ecc71", "GS": "#9b59b6", "FPar": "#f1c40f"}

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
            markers={"Base": "o", "GS": "^", "FPar": "s"},
            s=100,
            palette=type_palette,
            ax=ax,
        )

        ax.set_title(f"{mode_names[i]} Models: Efficiency")
        ax.set_xlabel("Model Parameters (Billions)")
        ax.set_xticks(list(size_map.values()))
        ax.set_xticklabels(list(size_map.keys()))

        if i == 0:
            ax.set_ylabel("Efficiency (F1 / kWh)")
        else:
            ax.set_ylabel("")

        ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig("efficiency_comparison.svg", bbox_inches="tight")
    print("Generated efficiency_comparison.svg")

    # ---------------------------------------------------------
    # PLOT 5: Performance-Energy Trade-off Scatter Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df["f1"],
        df["energy_kWh_per_question"],
        c=df["params_B"],
        s=100,
        alpha=0.7,
        cmap="viridis",
    )
    plt.colorbar(scatter, label="Model Size (B)")
    plt.xlabel("F1 Score")
    plt.ylabel("Energy Consumption (kWh/question)")
    plt.title("Performance vs Energy Trade-off")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("performance_energy_tradeoff.svg", bbox_inches="tight")
    print("Generated performance_energy_tradeoff.svg")

    # ---------------------------------------------------------
    # PLOT 6: Token Efficiency Analysis
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for thinking in [False, True]:
        subset = df[df["thinking"] == thinking]
        plt.plot(
            subset["params_B"],
            subset["tokens_per_f1"],
            marker="o",
            label=f"Thinking: {thinking}",
            linewidth=2,
        )
    plt.xlabel("Model Size (B)")
    plt.ylabel("Tokens per F1 Point")
    plt.title("Token Efficiency vs Model Size")
    plt.legend()
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("token_efficiency.svg", bbox_inches="tight")
    print("Generated token_efficiency.svg")

    # ---------------------------------------------------------
    # PLOT 7: RAG vs Base Model Comparison
    # ---------------------------------------------------------
    rag_comparison = (
        df.groupby(["params_B", "context_used"])
        .agg(
            {
                "f1": "max",
                "energy_kWh_per_question": "mean",
                "time_s_per_question": "mean",
            }
        )
        .reset_index()
    )

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    for context in [False, True]:
        subset = rag_comparison[rag_comparison["context_used"] == context]
        plt.plot(
            subset["params_B"],
            subset["f1"],
            marker="o",
            label="RAG" if context else "Base",
            linewidth=2,
        )
    plt.xlabel("Model Size (B)")
    plt.ylabel("Max F1 Score")
    plt.legend()
    plt.title("Performance: RAG vs Base")

    plt.subplot(1, 3, 2)
    for context in [False, True]:
        subset = rag_comparison[rag_comparison["context_used"] == context]
        plt.plot(
            subset["params_B"],
            subset["energy_kWh_per_question"],
            marker="o",
            label="RAG" if context else "Base",
            linewidth=2,
        )
    plt.xlabel("Model Size (B)")
    plt.ylabel("Energy (kWh/question)")
    plt.legend()
    plt.title("Energy: RAG vs Base")
    plt.yscale("log")

    plt.subplot(1, 3, 3)
    for context in [False, True]:
        subset = rag_comparison[rag_comparison["context_used"] == context]
        plt.plot(
            subset["params_B"],
            subset["time_s_per_question"],
            marker="o",
            label="RAG" if context else "Base",
            linewidth=2,
        )
    plt.xlabel("Model Size (B)")
    plt.ylabel("Time (s/question)")
    plt.legend()
    plt.title("Time: RAG vs Base")

    plt.tight_layout()
    plt.savefig("rag_vs_base_comparison.svg", bbox_inches="tight")
    print("Generated rag_vs_base_comparison.svg")

    # ---------------------------------------------------------
    # PLOT 8: Carbon Emissions Analysis
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for thinking in [False, True]:
        for context in [False, True]:
            subset = df[(df["thinking"] == thinking) & (df["context_used"] == context)]
            if len(subset) > 0:
                label = f"Thinking:{thinking}, RAG:{context}"
                plt.plot(
                    subset["params_B"],
                    subset["emissions_kg_per_question"] * 1000,
                    marker="o",
                    label=label,
                    linewidth=2,
                )
    plt.xlabel("Model Size (B)")
    plt.ylabel("CO2 Emissions (g/question)")
    plt.title("Carbon Emissions per Question")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("carbon_emissions.svg", bbox_inches="tight")
    print("Generated carbon_emissions.svg")

    # ---------------------------------------------------------
    # PLOT 9: Cost-Benefit Analysis: Performance per Unit Resources
    # ---------------------------------------------------------
    metrics = [
        "performance_per_token",
        "performance_per_second",
        "performance_per_emission",
    ]
    titles = ["F1 per Token", "F1 per Second", "F1 per kg CO2"]

    plt.figure(figsize=(15, 5))
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        plt.subplot(1, 3, i + 1)
        for thinking in [False, True]:
            subset = df[df["thinking"] == thinking]
            plt.plot(
                subset["params_B"],
                subset[metric],
                marker="o",
                label=f"Thinking: {thinking}",
                linewidth=2,
            )
        plt.xlabel("Model Size (B)")
        plt.ylabel(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        if metric != "performance_per_token":
            plt.yscale("log")

    plt.tight_layout()
    plt.savefig("resource_efficiency.svg", bbox_inches="tight")
    print("Generated resource_efficiency.svg")

    # ---------------------------------------------------------
    # PLOT 10: Thinking Mode Impact Analysis
    # ---------------------------------------------------------
    thinking_effect = []
    for size in df["params_B"].unique():
        base_subset = df[(df["params_B"] == size) & (df["thinking"] == False)]
        think_subset = df[(df["params_B"] == size) & (df["thinking"] == True)]

        if len(base_subset) > 0 and len(think_subset) > 0:
            base_f1 = base_subset["f1"].max()
            think_f1 = think_subset["f1"].max()
            improvement = ((think_f1 - base_f1) / base_f1) * 100 if base_f1 > 0 else 0

            base_energy = base_subset["energy_kWh_per_question"].mean()
            think_energy = think_subset["energy_kWh_per_question"].mean()
            energy_increase = ((think_energy - base_energy) / base_energy) * 100

            thinking_effect.append(
                {
                    "params_B": size,
                    "f1_improvement_pct": improvement,
                    "energy_increase_pct": energy_increase,
                    "improvement_per_energy": (
                        improvement / energy_increase if energy_increase != 0 else 0
                    ),
                }
            )

    thinking_df = pd.DataFrame(thinking_effect)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(thinking_df["params_B"], thinking_df["f1_improvement_pct"])
    plt.xlabel("Model Size (B)")
    plt.ylabel("F1 Improvement (%)")
    plt.title("Thinking Mode: F1 Improvement")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.bar(thinking_df["params_B"], thinking_df["energy_increase_pct"])
    plt.xlabel("Model Size (B)")
    plt.ylabel("Energy Increase (%)")
    plt.title("Thinking Mode: Energy Cost")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.bar(thinking_df["params_B"], thinking_df["improvement_per_energy"])
    plt.xlabel("Model Size (B)")
    plt.ylabel("Improvement per Energy Unit")
    plt.title("Thinking Mode Efficiency")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("thinking_impact_analysis.svg", bbox_inches="tight")
    print("Generated thinking_impact_analysis.svg")

    # ---------------------------------------------------------
    # PLOT 11: Pareto Frontier Analysis
    # ---------------------------------------------------------

    # Get Pareto frontier using the detailed function
    pareto_points = get_pareto_frontier_detailed(df, "energy_kWh_per_question", "f1")

    # Create the plot
    plt.figure(figsize=(14, 9))

    # Color code by thinking mode (same as original)
    colors = df["thinking"].map({True: "#ff6b6b", False: "#4ecdc4"})

    # Plot all points with color coding
    scatter = plt.scatter(
        df["energy_kWh_per_question"],
        df["f1"],
        c=colors,
        alpha=0.7,
        s=80,
        label="_nolegend_",
    )

    # Plot Pareto frontier line
    pareto_line = plt.plot(
        pareto_points[:, 0],
        pareto_points[:, 1],
        "k--",
        linewidth=2.5,
        alpha=0.8,
        label="Pareto Frontier",
    )

    # Plot Pareto optimal points with gold markers
    pareto_scatter = plt.scatter(
        pareto_points[:, 0],
        pareto_points[:, 1],
        c="gold",
        s=120,
        edgecolors="black",
        linewidth=1.5,
        label="Pareto Optimal",
        zorder=5,
    )

    # Define key configurations for special annotations
    key_configs = [
        "1.7-GS-NoThink",  # Best overall PPE
        "14-GS-NoThink",  # Highest F1 score
        "32-B-NoThink",  # Largest model baseline
    ]

    # Annotate Pareto optimal points EXCEPT key configs (they will be annotated separately)
    for point in pareto_points:
        # Skip if this is a key configuration (will be annotated separately)
        if point[2] in key_configs:
            continue

        plt.annotate(
            point[2],
            (point[0], point[1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            arrowprops=dict(
                arrowstyle="->", connectionstyle="arc3,rad=0", color="black", lw=0.8
            ),
        )

    # Highlight the key configurations with larger annotations - POSITIONED BOTTOM RIGHT
    for config in key_configs:
        point_data = df[df["config_label"] == config]
        if not point_data.empty:
            x_pos = point_data["energy_kWh_per_question"].values[0]
            y_pos = point_data["f1"].values[0]

            plt.annotate(
                config,
                (x_pos, y_pos),
                xytext=(15, -25),  # Bottom right positioning
                textcoords="offset points",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="yellow",
                    alpha=0.9,
                    edgecolor="black",
                    linewidth=1.5,
                ),
                arrowprops=dict(
                    arrowstyle="->",
                    connectionstyle="arc3,rad=-0.2",  # Negative rad for bottom placement
                    color="red",
                    lw=1.5,
                ),
                ha="left",
                va="top",
            )

    # Customize the plot
    plt.xlabel("Energy Consumption (kWh per question)", fontsize=13, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=13, fontweight="bold")
    plt.title(
        "Pareto Frontier: F1 Score vs. Energy Consumption\nQwen3 Model Family on Natural Questions",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # Set logarithmic scale for x-axis to better visualize the spread
    plt.xscale("log")

    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle="--")

    # Create a comprehensive legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#4ecdc4",
            markersize=10,
            label="Non-Thinking",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#ff6b6b",
            markersize=10,
            label="Thinking",
        ),
        Line2D(
            [0], [0], color="k", linestyle="--", linewidth=2.5, label="Pareto Frontier"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=10,
            label="Pareto Optimal",
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
        config_data = df[df["config_label"] == row["config"]].iloc[0]
        print(
            f"{row['config']:20} | F1: {row['f1']:.4f} | Energy: {row['energy']:.6f} kWh | PPE: {row['f1'] / row['energy']:.1f}"
        )

    # Additional analysis
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)
    best_small = df[df["config_label"] == "1.7-GS-NoThink"].iloc[0]
    best_large = df[df["config_label"] == "32-B-NoThink"].iloc[0]

    print(f"Best Small (1.7-GS-NoThink):")
    print(
        f"  F1: {best_small['f1']:.4f}, Energy: {best_small['energy_kWh_per_question']:.6f} kWh, PPE: {best_small['f1'] / best_small['energy_kWh_per_question']:.1f}"
    )
    print(f"Best Large (32-B-NoThink):")
    print(
        f"  F1: {best_large['f1']:.4f}, Energy: {best_large['energy_kWh_per_question']:.6f} kWh, PPE: {best_large['f1'] / best_large['energy_kWh_per_question']:.1f}"
    )
    print(
        f"Improvement: F1 +{((best_small['f1'] / best_large['f1']) - 1) * 100:.1f}%, PPE +{((best_small['f1'] / best_small['energy_kWh_per_question']) / (best_large['f1'] / best_large['energy_kWh_per_question']) - 1) * 100:.1f}%"
    )

    # ---------------------------------------------------------
    # TABLE: Best Small vs Largest
    # ---------------------------------------------------------
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
        "energy_kWh_per_question",
        "perf_per_energy",
    ]
    final_table = comparison_table[cols_to_show].sort_values("params_B")

    final_table.to_csv("best_config_comparison.csv", index=False)

    print("\n--- Best Configurations Table ---")
    print(final_table.to_string(index=False))

    print("\n=== All plots generated successfully! ===")


if __name__ == "__main__":
    main()
