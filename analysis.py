import pandas as pd
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# Load the CSV result files
distilgpt2_q = pd.read_csv("hotpot_distilgpt2_q.csv")
distilgpt2_qr = pd.read_csv("hotpot_distilgpt2_q+r.csv")
gpt2xl_q = pd.read_csv("hotpot_gpt2-xl_q.csv")

# Compute average metrics for each setup
summary = pd.DataFrame({
    "Model": ["distilgpt2_q", "distilgpt2_q+r", "gpt2-xl_q"],
    "Context Used": [False, True, False],
    "EM": [
        distilgpt2_q["em"].mean(),
        distilgpt2_qr["em"].mean(),
        gpt2xl_q["em"].mean()
    ],
    "F1": [
        distilgpt2_q["f1"].mean(),
        distilgpt2_qr["f1"].mean(),
        gpt2xl_q["f1"].mean()
    ],
    "Avg Energy (kWh)": [
        distilgpt2_q["energy_kWh"].mean(),
        distilgpt2_qr["energy_kWh"].mean(),
        gpt2xl_q["energy_kWh"].mean()
    ],
    "Avg Emissions (kg)": [
        distilgpt2_q["emissions (kg)"].mean(),
        distilgpt2_qr["emissions (kg)"].mean(),
        gpt2xl_q["emissions (kg)"].mean()
    ],
    "Avg Time (s)": [
        distilgpt2_q["time (s)"].mean(),
        distilgpt2_qr["time (s)"].mean(),
        gpt2xl_q["time (s)"].mean()
    ]
})

print(summary)
