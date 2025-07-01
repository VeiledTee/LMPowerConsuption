from pathlib import Path
import pandas as pd
import os
import smtplib
import mimetypes
from email.message import EmailMessage

from config import CONFIG
from utils import convert_seconds
import re


# Optional filter: set to a substring to include only matching files - set to None to include all
FILTER_SUBSTRING: str | None = "deepseek"

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
    "deepseek-r1-7b_q": "DeepSeek-r1 7B (Base)",
    "deepseek-r1-7b_q+r": "DeepSeek-r1 7B (RAG)",
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


def extract_model_size(display_name: str) -> float:
    match = re.search(r'(\d+(\.\d+)?)B', display_name, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df


def _combined(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    return df[c1] + df[c2]


def add_combined_cols(df: pd.DataFrame) -> pd.DataFrame:
    for new_name, (c1, c2) in RESULT_COLS.items():
        df[f"combined_{new_name}"] = _combined(df, c1, c2)
    return df


def summarise(
        model_key: str,
        path: Path,
        context_used: bool,
        dataset_version: str | int,
) -> dict:
    df = _load(path)
    df = add_combined_cols(df)
    display_name = MODEL_DISPLAY_NAMES[f"{model_key}_q+r"] if context_used else MODEL_DISPLAY_NAMES[f"{model_key}_q"]

    total_time_seconds = df["combined_time"].sum()
    hours, minutes, seconds = convert_seconds(total_time_seconds)

    return {
        "model": display_name,
        "context_used": context_used,
        "dataset": "BoolQ" if "boolq" in path.name.lower() else "HotpotQA",
        "dataset_version": str(dataset_version),
        "f1": df["f1"].mean(),
        "em": df["em"].mean(),
        "total_energy_kWh": df["combined_energy"].mean(),
        "inference_energy_kWh": df["inference_energy_consumed (kWh)"].mean(),
        "retrieval_energy_kWh": df["retrieval_energy_consumed (kWh)"].mean(),
        "total_emissions_kg": df["combined_emissions"].mean(),
        "inference_emissions_kg": df["inference_emissions (kg)"].mean(),
        "retrieval_emissions_kg": df["retrieval_emissions (kg)"].mean(),
        "avg_time_s": df["combined_time"].mean(),  # Average per question
        "total_time": f"{hours}:{minutes}:{seconds}"
    }


def send_email_with_attachment(from_addr: str, to_addr: str, subject: str, body: str, attachment_path: str):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to_addr
    msg.set_content(body)

    with open(attachment_path, 'rb') as f:
        data = f.read()
    maintype, subtype = (mimetypes.guess_type(attachment_path)[0] or 'application/octet-stream').split('/', 1)
    msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=os.path.basename(attachment_path))

    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = CONFIG.from_email
    smtp_pass = ""
    if not (smtp_user and smtp_pass):
        raise RuntimeError("SMTP_USERNAME and SMTP_PASSWORD must be set")

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    results_dir = project_dir / "results"

    files = sorted(results_dir.glob("*.csv"))
    if FILTER_SUBSTRING:
        files = [f for f in files if FILTER_SUBSTRING in f.name]

    summaries = []
    for csv_path in files:
        name = csv_path.stem
        parts = name.split('_')
        model_key = parts[1] if len(parts) == 3 else parts[2]
        context_used = 'q+r' in name
        dataset_version = 'full' if len(parts) == 3 else parts[1]
        summaries.append(
            summarise(
                model_key=model_key,
                path=csv_path,
                context_used=context_used,
                dataset_version=dataset_version,
            )
        )

    df_summary = pd.DataFrame(summaries)
    df_summary["model_size_b"] = df_summary["model"].apply(extract_model_size)
    df_summary["is_rag"] = df_summary["context_used"].astype(int)  # Base = 0, RAG = 1
    df_summary = df_summary.sort_values(by=["dataset_version", "model_size_b", "is_rag"])
    df_summary.drop(columns=["is_rag"], inplace=True)
    df_summary.drop(columns=["model_size_b"], inplace=True)

    out_csv = results_dir / "summary_results.csv"
    out_md = results_dir / "summary_results.md"
    df_summary.to_csv(out_csv, index=False, float_format="%.6f")

    print(df_summary.to_markdown(index=False, floatfmt=".6f"))
    with open(out_md, 'w') as f:
        f.write(df_summary.to_markdown(index=False, floatfmt=".6f"))

    print(f"Saved summary to {out_csv} and {out_md}")

    if CONFIG.email_results:
        send_email_with_attachment(
            from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
            to_addr=CONFIG.to_email,
            subject="Batch Analysis Summary",
            body="See attached summary CSV.",
            attachment_path=str(out_csv),
        )
        print("Email sent.")


if __name__ == "__main__":
    main()
