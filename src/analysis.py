import argparse
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

RESULT_COLS = {
    "energy": ("inference_energy_consumed (kWh)", "retrieval_energy_consumed (kWh)"),
    "emissions": ("inference_emissions (kg)", "retrieval_emissions (kg)"),
    "time": ("inference_duration (s)", "retrieval_duration (s)"),
}


def _combined_mean(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    """Average the inference + r columns."""
    return (df[c1] + df[c2]) / 2.0


def add_combined_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add energy_kWh, emissions_kg, time_s averaged over r + inference."""
    for new_name, (c1, c2) in RESULT_COLS.items():
        df[f"combined_{new_name}"] = _combined_mean(df, c1, c2)
    return df


def summarise(model_name: str, df: pd.DataFrame, context_used: bool) -> dict:
    """Return one-row summary dict for a single model run."""
    df = add_combined_cols(df)

    return {
        "model": model_name,
        "context_used": context_used,
        "em": df["em"].mean(),
        "f1": df["f1"].mean(),
        "avg_energy_kWh": df["combined_energy"].mean(),
        "avg_r_energy_kWh": df["retrieval_energy_consumed (kWh)"].mean(),
        "avg_i_energy_kWh": df["inference_energy_consumed (kWh)"].mean(),
        "avg_emissions_kg": df["combined_emissions"].mean(),
        "avg_r_emissions_kg": df["retrieval_emissions (kg)"].mean(),
        "avg_i_emissions_kg": df["inference_emissions (kg)"].mean(),
        "avg_time_s": df["combined_time"].mean(),
        "avg_r_time_s": df["retrieval_duration (s)"].mean(),
        "avg_i_time_s": df["inference_duration (s)"].mean(),
    }


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    results_dir = project_dir / "results"

    summaries = [
        summarise("distilgpt2_q", _load(results_dir / "hotpot_distilgpt2_q.csv"), False),
        summarise("distilgpt2_q+r", _load(results_dir / "hotpot_distilgpt2_q+r.csv"), True),
        summarise("gpt2-xl_q", _load(results_dir / "hotpot_gpt2-xl_q.csv"), False),
        summarise("gemma-2b-it_q", _load(results_dir / "hotpot_gemma-2b-it_q.csv"), False),
        summarise("gemma-2b-it_q+r", _load(results_dir / "hotpot_gemma-2b-it_q+r.csv"), True),
        summarise("gemma-7b-it_q", _load(results_dir / "hotpot_gemma-7b-it_q.csv"), False),
    ]
    print(pd.DataFrame(summaries).to_markdown(index=False))


if __name__ == "__main__":
    main()
