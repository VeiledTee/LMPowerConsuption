import pandas as pd

path = r"C:\Users\Ethan\Documents\PhD\LMPowerConsuption\emissions\emissions.csv"

df = pd.read_csv(path)

ci = df["emissions"] / df["energy_consumed"].replace(0, pd.NA)

print("Average carbon intensity:", ci.mean())
print(ci.max(), ci.min())
print("Unique carbon intensities:", ci.nunique())
