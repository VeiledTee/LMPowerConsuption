import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np


def parse_model_size(model_name):
    """Extracts model size in Billions (B) from string."""
    match = re.search(r'(\d+\.?\d*)B', str(model_name))
    if match:
        return float(match.group(1))
    return 0.0


def map_dataset_type(version):
    """Maps dataset version to simplified types for plotting."""
    version = str(version).lower()
    if 'question only' in version:
        return 'Base'
    elif 'gs paragraph' in version:
        return 'GS'
    elif 'first paragraph' in version:
        return 'FPar'
    return 'Other'


def get_marker(dtype):
    """Returns marker style based on dataset type."""
    if dtype == 'Base': return 'o'  # Circle
    if dtype == 'GS': return '^'  # Triangle
    if dtype == 'FPar': return 's'  # Square
    return 'x'


def main():
    # 1. Load Data
    try:
        df = pd.read_csv('qwen3_summary.csv')
    except FileNotFoundError:
        print("Error: 'qwen3_summary.csv' not found. Please ensure the file is in the same directory.")
        return

    # 2. Preprocessing
    # Extract Model Size
    df['params_B'] = df['model'].apply(parse_model_size)

    # Normalize Dataset Version Types
    df['type'] = df['dataset_version'].apply(map_dataset_type)

    # Calculate Performance per Energy (F1 / kWh)
    # Avoid division by zero
    df['energy_kWh_per_question'] = df['energy_kWh_per_question'].replace(0, np.nan)
    df['perf_per_energy'] = df['f1'] / df['energy_kWh_per_question']

    # Set general style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})

    # ---------------------------------------------------------
    # PLOT 1: Performance vs Model Size
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Filter to get the best performance per model size/thinking to avoid clutter
    # (or plot all points if distinct enough. Let's plot all to see distribution)
    sns.lineplot(
        data=df,
        x='params_B',
        y='f1',
        hue='thinking',
        style='thinking',
        markers=True,
        dashes=False,
        palette=['#e74c3c', '#3498db']  # Red/Blue
    )

    plt.title('Performance vs Model Size: F1 Score')
    plt.xlabel('Model Parameters (Billions)')
    plt.ylabel('F1 Score')
    plt.xscale('log')  # Log scale often helps with 0.6B vs 70B
    plt.xticks(df['params_B'].unique(), df['params_B'].unique())  # Force specific ticks
    plt.legend(title='Thinking Mode')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('plot_1_performance_vs_size.svg', bbox_inches='tight')
    print("Generated plot_1_performance_vs_size.svg")

    # ---------------------------------------------------------
    # PLOT 2: Energy vs Model Size
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))

    sns.lineplot(
        data=df,
        x='params_B',
        y='energy_kWh_per_question',
        hue='thinking',
        style='thinking',
        markers=True,
        dashes=False,
        palette=['#e74c3c', '#3498db']
    )

    plt.title('Energy Consumption vs Model Size')
    plt.xlabel('Model Parameters (Billions)')
    plt.ylabel('Energy (kWh per Question)')
    plt.xscale('log')
    plt.yscale('log')  # Energy often scales exponentially
    plt.xticks(df['params_B'].unique(), df['params_B'].unique())
    plt.legend(title='Thinking Mode')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig('plot_2_energy_vs_size.svg', bbox_inches='tight')
    print("Generated plot_2_energy_vs_size.svg")

    # ---------------------------------------------------------
    # PLOT 3 & 4: Performance per Energy (Split by Thinking)
    # ---------------------------------------------------------
    # Define custom markers manually for the scatterplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    modes = [False, True]
    mode_names = ["Non-Thinking", "Thinking"]

    # Define color palette for types to ensure consistency
    type_palette = {'Base': '#2ecc71', 'GS': '#9b59b6', 'FPar': '#f1c40f'}

    for i, mode in enumerate(modes):
        subset = df[df['thinking'] == mode]
        ax = axes[i]

        # We use scatterplot to control markers specifically
        sns.scatterplot(
            data=subset,
            x='params_B',
            y='perf_per_energy',
            hue='type',
            style='type',
            markers={'Base': 'o', 'GS': '^', 'FPar': 's'},
            s=100,  # size
            palette=type_palette,
            ax=ax
        )

        ax.set_title(f'{mode_names[i]} Models: Efficiency')
        ax.set_xlabel('Model Parameters (Billions)')
        ax.set_xscale('log')
        ax.set_xticks(df['params_B'].unique())
        ax.set_xticklabels(df['params_B'].unique())

        if i == 0:
            ax.set_ylabel('Efficiency (F1 / kWh)')
        else:
            ax.set_ylabel('')

        ax.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig('plot_3_efficiency_comparison.svg', bbox_inches='tight')
    print("Generated plot_3_efficiency_comparison.svg")

    # ---------------------------------------------------------
    # TABLE: Best Small vs Largest
    # ---------------------------------------------------------
    # Define "Small" as the smallest 2 sizes available, and "Large" as the largest size
    unique_sizes = sorted(df['params_B'].unique())

    if len(unique_sizes) < 2:
        print("Not enough model sizes to compare small vs large.")
        return

    small_sizes = unique_sizes[:2]  # Take smallest 2
    large_size = unique_sizes[-1]  # Take largest

    print(f"\nComparing Small Models ({small_sizes}B) vs Large Model ({large_size}B)")

    # Filter data
    small_df = df[df['params_B'].isin(small_sizes)].copy()
    large_df = df[df['params_B'] == large_size].copy()

    # Find BEST config for each small size (based on F1)
    # We want 1 best row per small model size
    best_small = small_df.loc[small_df.groupby('params_B')['f1'].idxmax()]

    # Take both largest configs (Usually Base and RAG/Thinking variations)
    # If there are many, we take the top 2 by F1
    best_large = large_df.sort_values('f1', ascending=False).head(2)

    # Combine
    comparison_table = pd.concat([best_small, best_large])

    # Select readable columns
    cols_to_show = ['model', 'type', 'thinking', 'params_B', 'f1', 'energy_kWh_per_question', 'perf_per_energy']
    final_table = comparison_table[cols_to_show].sort_values('params_B')

    # Save to CSV
    final_table.to_csv('best_config_comparison.csv', index=False)

    print("\n--- Best Configurations Table ---")
    print(final_table.to_string(index=False))


if __name__ == "__main__":
    main()
