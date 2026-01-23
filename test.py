import numpy as np
import matplotlib.pyplot as plt

# x^2 curve up to x=1
x = np.linspace(0, 1, 400)
y = x**2

plt.figure(figsize=(6, 6))

# --- FILLING THE SECTIONS ---
# Top Right (Red)
plt.fill_between([1, 2], 1, 2, color='red', alpha=0.3)
# Bottom Right (Green)
plt.fill_between([1, 2], 0, 1, color='green', alpha=0.3)
# Left Above Curve (Orange)
plt.fill_between(x, y, 2, color='orange', alpha=0.3)
# Area Under Curve (Blue)
plt.fill_between(x, 0, y, color='blue', alpha=0.3)

# --- ADDING LABELS ---
# Syntax: plt.text(x, y, "label", fontsize, ha='center', va='center')
plt.text(0.4, 1.2, "Bad", fontsize=20, ha='center', va='center') # Orange section
plt.text(0.7, 0.2, "Good", fontsize=20, ha='center', va='center') # Blue section
plt.text(1.5, 1.5, "Power-Hungry", fontsize=20, ha='center', va='center') # Red section
plt.text(1.5, 0.5, "Great", fontsize=20, ha='center', va='center') # Green section

# --- PLOTTING LINES AND MARKS ---
plt.plot(x, y, color='C0')
x_marks = [0.25, 0.5, 0.75, 1.0]
plt.scatter(x_marks, [xm**2 for xm in x_marks], color='C0')

plt.axvline(x=1, linestyle="-", color='C0')
plt.hlines(y=1, xmin=1, xmax=2, linestyles="-", color='C0')

# Limits and Formatting
plt.xlim(0, 2)
plt.ylim(0, 2)
plt.xticks([])
plt.yticks([])
plt.xlabel("F1 Score")
plt.ylabel("Model Emissions (g of CO$_2$)")

plt.savefig('Objective Space Partitioning and Pareto Frontier Analysis.pdf')