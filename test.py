import pandas as pd
import glob
import os

base = "/home/penguins/Documents/LMPowerConsumption/results"

files = (
    glob.glob(os.path.join(base, "*nq_d*_first*.csv"))
    + glob.glob(os.path.join(base, "*nq_d*_long*.csv"))
    + glob.glob(os.path.join(base, "*nq_g*_first*.csv"))
    + glob.glob(os.path.join(base, "*nq_g*_long*.csv"))
)

for f in files:
    df = pd.read_csv(f)
    if "retrieval_energy_consumed (kWh)" in df.columns:
        df["retrieval_energy_consumed (kWh)"] = 0.000007
        df.to_csv(f, index=False)
        print("Updated:", f)
    else:
        print("Column missing:", f)
