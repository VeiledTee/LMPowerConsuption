import argparse
from pathlib import Path

import pandas as pd


RESULT_COLS = {
    "energy": ("inference_energy_consumed (kWh)", "retrieval_energy_consumed (kWh)"),
    "emissions": ("inference_emissions (kg)", "retrieval_emissions (kg)"),
    "time": ("inference_duration (s)", "retrieval_duration (s)"),
}


def _combined(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    """Average the inference + retrieval columns."""
    return df[c1] + df[c2]


def add_combined_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add energy_kWh, emissions_kg, time_s averaged over r + inference."""
    for new_name, (c1, c2) in RESULT_COLS.items():
        df[f"combined_{new_name}"] = _combined(df, c1, c2)
    return df


def summarise(model_name: str, df: pd.DataFrame, context_used: bool, dataset_version) -> dict:
    """Return one-row summary dict for a single model run."""
    df = add_combined_cols(df)

    # Per-sample carbon intensities
    inf_intensity = (df["inference_emissions (kg)"] / df["inference_energy_consumed (kWh)"]).mean()
    ret_intensity = (df["retrieval_emissions (kg)"] / df["retrieval_energy_consumed (kWh)"]).mean()
    comb_intensity = (df["combined_emissions"] / df["combined_energy"]).mean()
    return {
        "model": model_name,
        "context_used": context_used,
        "dataset_version": str(dataset_version),
        "f1": df["f1"].mean(),
        "em": df["em"].mean(),

        # Total energy (correct aggregates)
        "avg_energy_kWh": df["combined_energy"].mean(),
        "inference_energy_kWh": df["inference_energy_consumed (kWh)"].mean(),
        "retrieval_energy_kWh": df["retrieval_energy_consumed (kWh)"].mean(),
        # Total emissions (correct aggregates)
        "avg_emissions_kg": df["combined_emissions"].mean(),
        "inference_emissions_kg": df["inference_emissions (kg)"].mean(),
        "retrieval_emissions_kg": df["retrieval_emissions (kg)"].mean(),

        # Mean carbon intensities (true per-sample means)
        "mean_i_carbon_intensity": inf_intensity,
        "mean_r_carbon_intensity": ret_intensity,
        "mean_combined_carbon_intensity": comb_intensity,

        # Total durations
        "avg_time_s": df["combined_time"].mean(),
    }


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    results_dir = project_dir / "results"

    summaries = [
        summarise("distilgpt2_q", _load(results_dir / "hotpot_distilgpt2_q.csv"), False, 'full'),
        summarise("distilgpt2_q+r", _load(results_dir / "hotpot_distilgpt2_q+r.csv"), True, 'full'),
        summarise("gpt2-xl_q", _load(results_dir / "hotpot_gpt2-xl_q.csv"), False, 'full'),

        summarise("distilgpt2_q", _load(results_dir / "hotpot_mini_128_distilgpt2_q.csv"), False, '128'),
        summarise("distilgpt2_q+r", _load(results_dir / "hotpot_mini_128_distilgpt2_q+r.csv"), True, '128'),
        summarise("gpt2-xl_q", _load(results_dir / "hotpot_mini_128_gpt2-xl_q.csv"), False, '128'),

        # summarise("distilgpt2_q", _load(results_dir / "hotpot_mini_512_distilgpt2_q.csv"), False, '512'),
        # summarise("distilgpt2_q+r", _load(results_dir / "hotpot_mini_512_distilgpt2_q+r.csv"), True, '512'),
        # summarise("gpt2-xl_q", _load(results_dir / "hotpot_mini_512_gpt2-xl_q.csv"), False, '512'),

        summarise("gemma-2b_q", _load(results_dir / "hotpot_mini_128_gemma-2b_q.csv"), False, '128'),
        summarise("gemma-2b_q+r", _load(results_dir / "hotpot_mini_128_gemma-2b_q+r.csv"), True, '128'),
        summarise("gemma-7b_q", _load(results_dir / "hotpot_mini_128_gemma-7b_q.csv"), False, '128'),

        summarise("gemma-2b-it_q", _load(results_dir / "hotpot_mini_128_gemma-2b-it_q.csv"), False, '128'),
        summarise("gemma-2b-it_q+r", _load(results_dir / "hotpot_mini_128_gemma-2b-it_q+r.csv"), True, '128'),
        summarise("gemma-7b-it_q", _load(results_dir / "hotpot_mini_128_gemma-7b-it_q.csv"), False, '128'),
    ]
    pd.DataFrame(summaries).to_csv('summaries.csv', index=False)
    print(pd.DataFrame(summaries).to_markdown(index=False))


if __name__ == "__main__":
    main()
