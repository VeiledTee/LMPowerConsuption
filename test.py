import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data
df = pd.read_csv("results/gemma3_summary.csv")

# 2. Data Preparation
# Convert emissions from kg per question to grams per question
df['emissions_g_per_question'] = df['emissions_kg_per_question'] * 1000

# Extract model size for better labeling (reusing your logic)
df['params_B'] = df['model'].str.extract(r'(\d+\.?\d*)B').astype(float)

# 3. Plotting
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Create the scatter plot
# X-axis: f1 score | Y-axis: emissions in grams
plot = sns.scatterplot(
    data=df,
    x='f1',
    y='emissions_g_per_question',
    hue='model',
    style='dataset_version',
    s=100,
    palette='viridis'
)

# 4. Formatting
plt.title("Model Efficiency: Emissions (g $CO_2$) vs. F1 Score", fontsize=14)
plt.xlabel("F1 Score (Accuracy)", fontsize=12)
plt.ylabel("Emissions per Question (g $CO_2$)", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Models & Versions")

plt.tight_layout()
plt.show()