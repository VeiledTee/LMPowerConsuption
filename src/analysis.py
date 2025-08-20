import mimetypes
import os
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path
import tiktoken
import argparse

import pandas as pd

from config import CONFIG
from utils import convert_seconds

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
    "gemma3-1b_q": "Gemma3 1B (Base)",
    "gemma3-1b_q+r": "Gemma3 1B (RAG)",
    "gemma3-4b_q": "Gemma3 4B (Base)",
    "gemma3-4b_q+r": "Gemma3 4B (RAG)",
    "gemma3-12b_q": "Gemma3 12B (Base)",
    "gemma3-12b_q+r": "Gemma3 12B (RAG)",
    "gemma3-27b_q": "Gemma3 27B (Base)",
    "gemma3-27b_q+r": "Gemma3 27B (RAG)",
}

RESULT_COLS = {
    "energy": ("inference_energy_consumed (kWh)", "retrieval_energy_consumed (kWh)"),
    "emissions": ("inference_emissions (kg)", "retrieval_emissions (kg)"),
    "time": ("inference_duration (s)", "retrieval_duration (s)"),
}


def extract_family(model: str) -> str:
    """Extracts the base model family name (e.g., 'Gemma')."""
    return model.split()[0]


def extract_size(model: str) -> float:
    """Extracts the model size in billions of parameters."""
    match = re.search(r"(\d+(?:\.\d+)?)B", model)
    return float(match.group(1)) if match else 0.0


def extract_rag_flag(model: str) -> int:
    """Returns 1 if '(RAG)' is in the model name, otherwise 0."""
    return int("(RAG)" in model)


def extract_model_family(name: str) -> str:
    if "Gemma3" in name:
        return "Gemma3"
    elif "Gemma" in name:
        return "Gemma"
    elif "DeepSeek" in name:
        return "DeepSeek"
    elif "GPT2" in name:
        return "GPT2"
    elif "DistilGPT2" in name:
        return "DistilGPT2"
    else:
        return "Other"


def extract_model_size(display_name: str) -> float:
    match = re.search(r"(\d+(\.\d+)?)B", display_name, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def count_tokens(text, encoding_name="cl100k_base"):
    """Counts the number of tokens in a text string."""
    # This handles potential non-string inputs gracefully
    if not isinstance(text, str):
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


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
    try:
        display_name = (
            MODEL_DISPLAY_NAMES[f"{model_key}_q+r"]
            if context_used
            else MODEL_DISPLAY_NAMES[f"{model_key}_q"]
        )
        display_name = display_name + " Think" if "think" in str(path) else display_name

        total_time_seconds = df["combined_time"].sum()
        hours, minutes, seconds = convert_seconds(total_time_seconds)

        avg_pred_tokens = df["original_pred"].apply(count_tokens).mean()

        return {
            "model": display_name,
            "context_used": context_used,
            "dataset": "BoolQ" if "boolq" in path.name.lower() else "HotpotQA",
            "dataset_version": str(dataset_version),
            "f1": df["f1"].mean(),
            "em": df["em"].mean(),
            "avg_pred_tokens": avg_pred_tokens,
            "energy_kWh_per_question": df["combined_energy"].mean(),
            "inference_energy_kWh": df["inference_energy_consumed (kWh)"].mean(),
            "retrieval_energy_kWh": df["retrieval_energy_consumed (kWh)"].mean(),
            "emissions_kg_per_question": df["combined_emissions"].mean(),
            "inference_emissions_kg": df["inference_emissions (kg)"].mean(),
            "retrieval_emissions_kg": df["retrieval_emissions (kg)"].mean(),
            "avg_time_s": df["combined_time"].mean(),  # Average per question
            "total_time": f"{hours}:{minutes:02}:{seconds:02}",
        }
    except KeyError:
        return {}


# Function to calculate stats
def emission_stats(df_subset, model_name):
    emissions = df_subset["total_emissions_kg"]
    mean = emissions.mean()
    std_dev = emissions.std()
    val_range = emissions.max() - emissions.min()
    cv = std_dev / mean if mean != 0 else float("inf")
    range_pct_mean = val_range / mean * 100 if mean != 0 else float("inf")

    stats_output = (
        f"Stats for {model_name}\n"
        f"Mean: {mean:.8f} kg\n"
        f"Standard Deviation: {std_dev:.8f} kg\n"
        f"Range: {val_range:.8f} kg\n"
        f"Coefficient of Variation: {cv:.2%}\n"
        f"Range as % of Mean: {range_pct_mean:.2f}%\n"
    )

    # Save to text file
    file_path = f"{CONFIG.result_dir}/energy_stats.txt"
    with open(file_path, "w") as f:
        f.write(stats_output)

    print(f"\nStats for {model_name}")
    print(f"Mean: {mean:.8f} kg")
    print(f"Standard Deviation: {std_dev:.8f} kg")
    print(f"Range: {val_range:.8f} kg")
    print(f"Coefficient of Variation: {cv:.2%}")
    print(f"Range as % of Mean: {range_pct_mean:.2f}%")


def send_email_with_attachment(
    from_addr: str, to_addr: str, subject: str, body: str, attachment_path: str
):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    with open(attachment_path, "rb") as f:
        data = f.read()
    maintype, subtype = (
        mimetypes.guess_type(attachment_path)[0] or "application/octet-stream"
    ).split("/", 1)
    msg.add_attachment(
        data,
        maintype=maintype,
        subtype=subtype,
        filename=os.path.basename(attachment_path),
    )

    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = CONFIG.from_email
    smtp_pass = CONFIG.smtp_password
    if not (smtp_user and smtp_pass):
        raise RuntimeError("SMTP_USERNAME and SMTP_PASSWORD must be set")

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


def run_summary(filter_substring: str | None = None) -> None:
    results_dir = CONFIG.result_dir

    files = sorted(results_dir.glob("*.csv"))
    if filter_substring:
        files = [f for f in files if filter_substring in f.name]

    out_filename: str = (
        f"{filter_substring.split('_')[-1] if filter_substring else ''}{'_' if filter_substring else ''}summary"
    )

    summaries = []
    for csv_path in files:
        name = csv_path.stem
        parts = name.split("_")
        model_key = parts[1]
        context_used = "q+r" in name
        dataset_version = ""

        if len(parts) == 3:
            dataset_version += "full"
        elif "think" in name and len(parts) == 4:
            dataset_version += "full"
        elif "128" in name or "dev" in name:
            dataset_version = f"dev ({parts[3]})"
        else:
            dataset_version = "full"

        cur_summary = summarise(
            model_key=model_key,
            path=csv_path,
            context_used=context_used,
            dataset_version=dataset_version,
        )
        if cur_summary != {}:
            summaries.append(cur_summary)
        else:
            continue

    df_summary = pd.DataFrame(summaries)
    df_summary["model_size_b"] = df_summary["model"].apply(extract_model_size)
    df_summary["is_rag"] = df_summary["context_used"].astype(int)  # Base = 0, RAG = 1
    df_summary = df_summary.sort_values(
        by=["dataset", "dataset_version", "model_size_b", "is_rag"]
    )
    df_summary.drop(columns=["is_rag", "model_size_b"], inplace=True)

    out_csv = results_dir / f"{out_filename}.csv"
    out_md = results_dir / f"{out_filename}.md"
    df_summary.to_csv(out_csv, index=False, float_format="%.6f")

    def insert_blank_lines(df: pd.DataFrame) -> str:
        df["model_family"] = df["model"].apply(extract_model_family)
        output = ""
        grouped = df.groupby(["model_family", "dataset", "dataset_version"], sort=False)
        for (family, _, _), group in grouped:
            output += f"### {family}\n\n"
            output += (
                group.drop(columns=["model_family"]).to_markdown(
                    index=False, floatfmt=".6f"
                )
                + "\n\n"
            )
        return output

    markdown_with_splits = insert_blank_lines(df_summary)
    print(markdown_with_splits)
    with open(out_md, "w") as f:
        f.write(markdown_with_splits)

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


def run_variance_check(input_file: str) -> None:
    """Loads the summary file and calculates emission variance stats for each model."""
    summary_file_path = CONFIG.result_dir / input_file
    print(f"Running emission variance check on: {summary_file_path}")

    df = pd.read_csv(summary_file_path)

    # Add sorting columns
    df["model_family"] = df["model"].apply(extract_family)
    df["model_size"] = df["model"].apply(extract_size)
    df["is_rag"] = df["model"].apply(extract_rag_flag)

    # Sort correctly
    df_sorted = df.sort_values(
        by=["model_family", "model_size", "is_rag"],
        ascending=[True, True, True],
    )

    # Calculate and print stats for each model
    for model_name, model_df in df_sorted.groupby("model", sort=False):
        emission_stats(
            model_df, model_name
        )  # Assuming emission_stats is defined elsewhere


if __name__ == "__main__":
    run_summary(filter_substring="_deepseek")
    # run_variance_check(input_file='_summary.csv')
