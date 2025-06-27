from pathlib import Path

import pandas as pd
import os
import argparse
import smtplib
import mimetypes
from email.message import EmailMessage

from src.config import CONFIG

# Mapping from internal model keys to display names
MODEL_DISPLAY_NAMES = {
    "distilgpt2_q": "DistilGPT2 (Base)",
    "distilgpt2_q+r": "DistilGPT2 (RAG)",
    "gpt2-xl_q": "GPT2-XL (Base)",
    "gemma-2b_q": "Gemma 2B (Base)",
    "gemma-2b_q+r": "Gemma 2B (RAG)",
    "gemma-7b_q": "Gemma 7B (Base)",
    "gemma-2b-it_q": "Gemma 2B-IT (Base)",
    "gemma-2b-it_q+r": "Gemma 2B-IT (RAG)",
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


def send_email_with_attachment(from_addr: str, to_addr: str, subject: str, body: str, attachment_path: str):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg.set_content(body)

    # Attach CSV
    with open(attachment_path, 'rb') as f:
        data = f.read()
    maintype, subtype = (mimetypes.guess_type(attachment_path)[0] or 'application/octet-stream').split('/', 1)
    msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(attachment_path))

    # Send via SMTP
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = CONFIG.from_email
    smtp_pass = ""
    if not (smtp_user and smtp_pass):
        raise RuntimeError("SMTP_USERNAME and SMTP_PASSWORD must be set in environment")

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


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
        # summarise(
        #     "distilgpt2_q", results_dir / "boolq_128_distilgpt2_q.csv", False, "128"
        # ),
        # summarise(
        #     "distilgpt2_q+r", results_dir / "boolq_128_distilgpt2_q+r.csv", True, "128"
        # ),
        # summarise("gpt2-xl_q", results_dir / "boolq_128_gpt2-xl_q.csv", False, "128"),
        # summarise(
        #     "gemma-2b-it_q", results_dir / "boolq_128_gemma-2b-it_q.csv", False, "128"
        # ),
        # summarise(
        #     "gemma-2b-it_q+r",
        #     results_dir / "boolq_128_gemma-2b-it_q+r.csv",
        #     True,
        #     "128",
        # ),
        # summarise(
        #     "gemma-7b-it_q", results_dir / "boolq_128_gemma-7b-it_q.csv", False, "128"
        # ),
        # summarise(
        #     "gemma-7b-it_q",
        #     results_dir / "boolq_128_gemma-7b-it_q_simplified.csv",
        #     False,
        #     "128",
        #     model_name="Gemma 7B-IT (Simplified)",
        # ),

        summarise(
            "deepseek-r1-1.5b_q",
            results_dir / "boolq_128_deepseek-r1-1.5b_q.csv",
            False,
            "128",
        ),
        summarise(
            "deepseek-r1-1.5b_q+r",
            results_dir / "boolq_128_deepseek-r1-1.5b_q+r.csv",
            True,
            "128",
        ),
        summarise(
            "deepseek-r1-8b_q",
            results_dir / "boolq_128_deepseek-r1-8b_q.csv",
            False,
            "128",
        ),
        summarise(
            "deepseek-r1-8b_q+r",
            results_dir / "boolq_128_deepseek-r1-8b_q+r.csv",
            True,
            "128",
        ),
        summarise(
            "deepseek-r1-14b_q",
            results_dir / "boolq_128_deepseek-r1-14b_q.csv",
            False,
            "128",
        ),
        summarise(
            "deepseek-r1-14b_q+r",
            results_dir / "boolq_128_deepseek-r1-14b_q+r.csv",
            True,
            "128",
        ),
    ]

    df_summary = pd.DataFrame(summaries)

    save_csv = results_dir / "boolq_summaries.csv"
    save_md = results_dir / "boolq_summaries.md"
    df_summary.to_csv(save_csv, index=False, float_format="%.6f")
    df_summary.to_markdown(save_md.open("w"), index=False, floatfmt=[".6f"] * len(df_summary.columns))

    print(f"Saved analysis to {save_csv} and {save_md}")

    if CONFIG.email_results:
        print(f"Emailing results to {CONFIG.to_email}…")
        send_email_with_attachment(
            from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
            to_addr=CONFIG.to_email,
            subject="LMPowerConsumption Summary Results",
            body="Please find attached the CSV summary of results.",
            attachment_path=str(save_csv),
        )
        print("Email sent.")

    print(f"\nSaved analysis to {save_csv} and {save_md}")

if __name__ == "__main__":
    main()
