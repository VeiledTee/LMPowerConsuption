from pathlib import Path

import pandas as pd

# Mapping from internal model keys to display names
MODEL_DISPLAY_NAMES = {
    "distilgpt2_q": "DistilGPT2 (Base)",
    "distilgpt2_q+r": "DistilGPT2 (Context)",
    "gpt2-xl_q": "GPT2-XL (Base)",
    "gemma-2b_q": "Gemma 2B (Base)",
    "gemma-2b_q+r": "Gemma 2B (Context)",
    "gemma-7b_q": "Gemma 7B (Base)",
    "gemma-2b-it_q": "Gemma 2B-IT (Base)",
    "gemma-2b-it_q+r": "Gemma 2B-IT (Context)",
    "gemma-7b-it_q": "Gemma 7B-IT (Base)",
    "deepseek-r1-1.5b_q": "DeepSeek-r1 1.5B (Base)",
    "deepseek-r1-1.5b_q+r": "DeepSeek-r1 1.5B (RAG)",
    "deepseek-r1-8b_q": "DeepSeek-r1 8B (Base)",
    "deepseek-r1-8b_q+r": "DeepSeek-r1 8B (RAG)",
    "deepseek-r1-14b_q": "DeepSeek-r1 14B (Base)",
    "deepseek-r1-14b_q+r": "DeepSeek-r1 14B (RAG)",
    "deepseek-r1-32b_q": "DeepSeek-r1 32B (Base)",
}

RESULT_COLS = {
    "energy": ("inference_energy_consumed (kWh)", "retrieval_energy_consumed (kWh)"),
    "emissions": ("inference_emissions (kg)", "retrieval_emissions (kg)"),
    "time": ("inference_duration (s)", "retrieval_duration (s)"),
}


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _combined(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    """Average the inference + retrieval columns."""
    return df[c1] + df[c2]


def add_combined_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined energy, emissions, and time by summing inference + retrieval."""
    for new_name, (c1, c2) in RESULT_COLS.items():
        df[f"combined_{new_name}"] = _combined(df, c1, c2)
    return df


def summarise(
    model_key: str,
    path: Path,
    context_used: bool,
    dataset_version: str | int,
    model_name: str | None = None,
) -> dict:
    """Return one-row summary dict for a single model run, using display names."""
    df = _load(path)
    df = add_combined_cols(df)
    display_name = (
        model_name if model_name else MODEL_DISPLAY_NAMES.get(model_key, model_key)
    )
    return {
        "model": display_name,
        "context_used": context_used,
        "dataset": "BoolQ" if "boolq" in str(path) else "HotpotQA",
        "dataset_version": str(dataset_version),
        "f1": df["f1"].mean(),
        "em": df["em"].mean(),
        # Energy
        "total_energy_kWh": df["combined_energy"].mean(),
        "inference_energy_kWh": df["inference_energy_consumed (kWh)"].mean(),
        "retrieval_energy_kWh": df["retrieval_energy_consumed (kWh)"].mean(),
        # Emissions
        "total_emissions_kg": df["combined_emissions"].mean(),
        "inference_emissions_kg": df["inference_emissions (kg)"].mean(),
        "retrieval_emissions_kg": df["retrieval_emissions (kg)"].mean(),
        # Time
        "total_time_s": df["combined_time"].mean(),
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    results_dir = project_dir / "results"

    summaries = [
        # summarise(
        #     "distilgpt2_q", results_dir / "hotpot_distilgpt2_q.csv", False, "full"
        # ),
        # summarise(
        #     "distilgpt2_q+r", results_dir / "hotpot_distilgpt2_q+r.csv", True, "full"
        # ),
        # summarise("gpt2-xl_q", results_dir / "hotpot_gpt2-xl_q.csv", False, "full"),
        # summarise(
        #     "distilgpt2_q",
        #     results_dir / "hotpot_mini_128_distilgpt2_q.csv",
        #     False,
        #     "128",
        # ),
        # summarise(
        #     "distilgpt2_q+r",
        #     results_dir / "hotpot_mini_128_distilgpt2_q+r.csv",
        #     True,
        #     "128",
        # ),
        # summarise(
        #     "gpt2-xl_q", results_dir / "hotpot_mini_128_gpt2-xl_q.csv", False, "128"
        # ),
        # summarise(
        #     "gemma-2b_q", results_dir / "hotpot_mini_128_gemma-2b_q.csv", False, "128"
        # ),
        # summarise(
        #     "gemma-2b_q+r",
        #     results_dir / "hotpot_mini_128_gemma-2b_q+r.csv",
        #     True,
        #     "128",
        # ),
        # summarise(
        #     "gemma-7b_q", results_dir / "hotpot_mini_128_gemma-7b_q.csv", False, "128"
        # ),
        # summarise(
        #     "gemma-2b-it_q",
        #     results_dir / "hotpot_mini_128_gemma-2b-it_q.csv",
        #     False,
        #     "128",
        # ),
        # summarise(
        #     "gemma-2b-it_q+r",
        #     results_dir / "hotpot_mini_128_gemma-2b-it_q+r.csv",
        #     True,
        #     "128",
        # ),
        # summarise(
        #     "gemma-7b-it_q",
        #     results_dir / "hotpot_mini_128_gemma-7b-it_q.csv",
        #     False,
        #     "128",
        # ),
        summarise(
            "distilgpt2_q", results_dir / "boolq_128_distilgpt2_q.csv", False, "128"
        ),
        summarise(
            "distilgpt2_q+r", results_dir / "boolq_128_distilgpt2_q+r.csv", True, "128"
        ),
        summarise("gpt2-xl_q", results_dir / "boolq_128_gpt2-xl_q.csv", False, "128"),
        summarise(
            "gemma-2b-it_q", results_dir / "boolq_128_gemma-2b-it_q.csv", False, "128"
        ),
        summarise(
            "gemma-2b-it_q+r",
            results_dir / "boolq_128_gemma-2b-it_q+r.csv",
            True,
            "128",
        ),
        summarise(
            "gemma-7b-it_q", results_dir / "boolq_128_gemma-7b-it_q.csv", False, "128"
        ),
        summarise(
            "gemma-7b-it_q",
            results_dir / "boolq_128_gemma-7b-it_q_simplified.csv",
            False,
            "128",
            model_name="Gemma 7B-IT (Simplified)",
        ),

        summarise(
            "deepseek-r1-1.5_q",
            results_dir / "boolq_128_deepseek-r1-1.5b_q.csv",
            False,
            "128",
        ),
        summarise(
            "deepseek-r1-1.5_q+r",
            results_dir / "boolq_128_deepseek-r1-1.5b_q+r.csv",
            True,
            "128",
        ),
        summarise(
            "deepseek-r1-8_q",
            results_dir / "boolq_128_deepseek-r1-8b_q.csv",
            False,
            "128",
        ),
        summarise(
            "deepseek-r1-8_q+r",
            results_dir / "boolq_128_deepseek-r1-8b_q+r.csv",
            True,
            "128",
        ),
        summarise(
            "deepseek-r1-14_q",
            results_dir / "boolq_128_deepseek-r1-14b_q.csv",
            False,
            "128",
        ),
        summarise(
            "deepseek-r1-14_q+r",
            results_dir / "boolq_128_deepseek-r1-14b_q+r.csv",
            True,
            "128",
        ),
    ]

    df_summary = pd.DataFrame(summaries)

    save_path = str(results_dir / "boolq_summaries")
    df_summary.to_csv(
        save_path + ".csv", index=False, float_format="%.6f"
    )
    print(df_summary.to_markdown(index=False, floatfmt=".6f"))
    with open(save_path + ".md", "w") as f:
        f.write(
            df_summary.to_markdown(
                index=False, floatfmt=[".6f"] * len(df_summary.columns)
            )
        )

    print(f"\nSaved analysis to {save_path}.csv and {save_path}.md")

if __name__ == "__main__":
    main()
