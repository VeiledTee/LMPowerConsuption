import mimetypes
import os
import re
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import tiktoken

from src.config import CONFIG
from src.utils import convert_seconds

# Mapping from internal model keys to display names
MODEL_DISPLAY_NAMES: Dict[str, str] = {
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
    "qwen3-0.6b_q": "Qwen3 0.6B (Base)",
    "qwen3-0.6b_q+r": "Qwen3 0.6B (RAG)",
    "qwen3-1.7b_q": "Qwen3 1.7B (Base)",
    "qwen3-1.7b_q+r": "Qwen3 1.7B (RAG)",
    "qwen3-4b_q": "Qwen3 4B (Base)",
    "qwen3-4b_q+r": "Qwen3 4B (RAG)",
    "qwen3-8b_q": "Qwen3 8B (Base)",
    "qwen3-8b_q+r": "Qwen3 8B (RAG)",
    "qwen3-14b_q": "Qwen3 14B (Base)",
    "qwen3-14b_q+r": "Qwen3 14B (RAG)",
    "qwen3-32b_q": "Qwen3 32B (Base)",
    "qwen3-32b_q+r": "Qwen3 32B (RAG)",
}

RESULT_COLS: Dict[str, Tuple[str, str]] = {
    "energy": ("inference_energy_consumed (kWh)", "retrieval_energy_consumed (kWh)"),
    "emissions": ("inference_emissions (kg)", "retrieval_emissions (kg)"),
    "time": ("inference_duration (s)", "retrieval_duration (s)"),
}


def extract_family(model: str) -> str:
    """Extracts the base model family name from a model display name.

    Args:
        model: The model display name (e.g., 'Gemma 2B (Base)')

    Returns:
        The extracted family name (e.g., 'Gemma')
    """
    return model.split()[0]


def extract_size(model: str) -> float:
    """Extracts the model size in billions of parameters from a model display name.

    Args:
        model: The model display name

    Returns:
        The model size in billions of parameters as a float, or 0.0 if not found
    """
    match = re.search(r"(\d+(?:\.\d+)?)B", model)
    return float(match.group(1)) if match else 0.0


def extract_rag_flag(model: str) -> int:
    """Determines if a model uses RAG based on its display name.

    Args:
        model: The model display name

    Returns:
        1 if '(RAG)' is in the model name, otherwise 0
    """
    return int("(RAG)" in model)


def extract_model_family(name: str) -> str:
    """Extracts the model family from a display name.

    Args:
        name: The model display name

    Returns:
        The model family name (Gemma3, Gemma, DeepSeek, GPT2, DistilGPT2, or Other)
    """
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
    elif "qwen3" in name:
        return "Qwen3"
    else:
        return "Other"


def extract_model_size(display_name: str) -> float:
    """Extracts the model size from a display name.

    Args:
        display_name: The model display name

    Returns:
        The model size in billions of parameters as a float, or 0.0 if not found
    """
    match = re.search(r"(\d+(\.\d+)?)B", display_name, re.IGNORECASE)
    return float(match.group(1)) if match else 0.0


def count_tokens(text: Union[str, object], encoding_name: str = "cl100k_base") -> int:
    """Counts the number of tokens in a text string.

    Args:
        text: The text to count tokens in (non-string inputs return 0)
        encoding_name: The encoding to use for token counting

    Returns:
        The number of tokens in the text
    """
    if not isinstance(text, str):
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def _load(path: Path) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame.

    Args:
        path: Path to the CSV file

    Returns:
        A pandas DataFrame containing the CSV data

    Raises:
        FileNotFoundError: If the file does not exist
    """
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _combined(df: pd.DataFrame, c1: str, c2: str) -> pd.Series:
    """Combines two columns from a DataFrame.

    Args:
        df: The DataFrame containing the columns
        c1: Name of the first column
        c2: Name of the second column

    Returns:
        A Series with the sum of the two columns
    """
    return df[c1] + df[c2]


def add_combined_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Adds combined columns for energy, emissions, and time to a DataFrame.

    Args:
        df: The input DataFrame

    Returns:
        The DataFrame with additional combined columns
    """
    for new_name, (c1, c2) in RESULT_COLS.items():
        df[f"combined_{new_name}"] = _combined(df, c1, c2)
    return df


def summarise(
    model_key: str,
    path: Path,
    context_used: bool,
    dataset_version: Union[str, int],
) -> Dict[str, Union[str, float, int]]:
    """Summarizes the results from a model evaluation CSV file.

    Args:
        model_key: The key identifying the model
        path: Path to the CSV file with evaluation results
        context_used: Whether context/retrieval was used
        dataset_version: Version of the dataset used

    Returns:
        A dictionary with summary statistics for the model evaluation
    """
    df = _load(path)
    df = add_combined_cols(df)

    # Get display name based on model key and context usage
    display_name = (
        MODEL_DISPLAY_NAMES[f"{model_key}_q+r"]
        if context_used
        else MODEL_DISPLAY_NAMES[f"{model_key}_q"]
    )

    # Add "Think" suffix if applicable
    # display_name = display_name + " Think" if "think" in str(path) else display_name

    # Calculate total time and convert to hours, minutes, seconds
    total_time_seconds = df["combined_time"].sum()
    hours, minutes, seconds = convert_seconds(total_time_seconds)

    # Calculate total energy
    total_energy_kWh = df["combined_energy"].sum()

    # Calculate total and average prediction tokens
    total_pred_tokens = df["original_pred"].astype(str).apply(count_tokens).sum()
    avg_pred_tokens = df["original_pred"].astype(str).apply(count_tokens).mean()

    if "hotpot" in path.name.lower():
        dataset_name = "HotpotQA"
    elif "boolq" in path.name.lower():
        dataset_name = "BoolQ"
    elif "squad" in path.name.lower():
        dataset_name = "SQuAD v1"
    elif "squad_v2" in path.name.lower():
        dataset_name = "SQuAD v2"
    elif "nq" in path.name.lower():
        dataset_name = "NQ"

    return {
        "model": display_name,
        "context_used": context_used,
        "thinking": True if "think" in str(path) else False,
        "dataset": dataset_name,
        "dataset_version": str(dataset_version),
        "f1": df["f1"].mean(),
        "em": df["em"].mean(),
        "pred_tokens_per_question": avg_pred_tokens,
        "total_tokens": total_pred_tokens,
        "total_energy_kWh": total_energy_kWh,
        "energy_kWh_per_question": df["combined_energy"].mean(),
        "inference_energy_kWh": df["inference_energy_consumed (kWh)"].mean(),
        "retrieval_energy_kWh": df["retrieval_energy_consumed (kWh)"].mean(),
        "emissions_kg_per_question": df["combined_emissions"].mean(),
        "inference_emissions_kg_per_question": df[
            "inference_emissions (kg)"
        ].mean(),
        "retrieval_emissions_kg_per_question": df[
            "retrieval_emissions (kg)"
        ].mean(),
        "time_s_per_question": df["combined_time"].mean(),
        "total_time": f"{hours}:{minutes:02}:{seconds:02}",
    }


def emission_stats(df_subset: pd.DataFrame, model_name: str) -> None:
    """Calculates and prints emission statistics for a model.

    Args:
        df_subset: DataFrame containing emission data for a specific model
        model_name: Name of the model for reporting
    """
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
) -> None:
    """Sends an email with an attachment.

    Args:
        from_addr: Sender email address
        to_addr: Recipient email address
        subject: Email subject
        body: Email body text
        attachment_path: Path to the file to attach

    Raises:
        RuntimeError: If SMTP credentials are not set
    """
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


def filter_files_by_dataset_version(
    files: List[Path], dataset_version: str
) -> List[Path]:
    """Filters files based on dataset version criteria.

    Args:
        files: List of file paths to filter
        dataset_version: Version of dataset to filter for

    Returns:
        Filtered list of file paths
    """
    files_to_process = []

    for f in files:
        name = f.stem
        # Determine dataset_version from filename
        if "dev (128)" in dataset_version:
            if "dev" in name or "128" in name:
                files_to_process.append(f)
        elif "full" in dataset_version:
            if "dev" not in name and "128" not in name:
                files_to_process.append(f)

    return files_to_process


def generate_output_filename(model_filter: Optional[str]) -> str:
    """Generates an output filename based on model filter.

    Args:
        model_filter: Optional filter string for models

    Returns:
        Base filename for output files
    """
    if model_filter:
        return f"{model_filter.split('_')[-1]}_summary"
    return "summary"


def determine_dataset_version(filename_parts: List[str]) -> str:
    """Determines the dataset version from filename parts.

    Args:
        filename_parts: List of parts from splitting the filename

    Returns:
        The dataset version string
    """
    if len(filename_parts) > 2 and (
        "dev" in filename_parts[2] or "128" in filename_parts[2]
    ):
        return f"dev ({filename_parts[3]})" if len(filename_parts) > 3 else "dev (128)"
    if filename_parts[0] == 'nq':
        if filename_parts[-1] == 'long':
            return 'GS Paragraph'
        elif filename_parts[-1] == 'first':
            return 'First Paragraph'
        else:
            return 'Question Only'
    return "full"


def insert_blank_lines(df: pd.DataFrame) -> str:
    """Formats a DataFrame as markdown with section headers for model families.

    Args:
        df: DataFrame to format

    Returns:
        Markdown string with formatted table and section headers
    """
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


def run_summary(
        model_filter: Optional[str] = None, dataset_version: Optional[str] = None
) -> None:
    """Main function to run summary analysis on evaluation results.

    Args:
        model_filter: Optional filter to select specific models
        dataset_version: Optional filter to select dataset version
    """
    results_dir = CONFIG.result_dir
    files = sorted(results_dir.glob("*.csv"))

    if model_filter:
        files = [f for f in files if model_filter in f.name]

    if dataset_version:
        files = filter_files_by_dataset_version(files, dataset_version)

    out_filename = generate_output_filename(model_filter)
    summaries = []

    for csv_path in files:
        name = csv_path.stem
        parts = name.split("_")
        model_key = parts[1]
        context_used = "q+r" in name
        current_dataset_version = determine_dataset_version(parts)

        cur_summary = summarise(
            model_key=model_key,
            path=csv_path,
            context_used=context_used,
            dataset_version=current_dataset_version,
        )

        if cur_summary:
            summaries.append(cur_summary)

    if not summaries:
        print("No valid summaries generated. Check your filters and data files.")
        return

    df_summary = pd.DataFrame(summaries)

    # Calculate 'Performance per Energy' (F1 divided by Total Energy in kWh)
    df_summary["performance_per_energy"] = (
            df_summary["f1"] / df_summary["total_energy_kWh"]
    )

    # Add model size and family for sorting
    df_summary["model_family"] = df_summary["model"].apply(extract_model_family)
    df_summary["model_size_b"] = df_summary["model"].apply(extract_model_size)

    # Sort by family, then size, then context
    df_summary = df_summary.sort_values(
        by=["model_family", "model_size_b", "context_used"]
    )
    # Drop the temporary columns used for sorting
    df_summary = df_summary.drop(columns=["model_family", "model_size_b"])

    # Print single dataframe
    print(df_summary.to_markdown(index=False, floatfmt=".6f"))

    # Save files
    out_csv = results_dir / f"{out_filename}.csv"
    out_md = results_dir / f"{out_filename}.md"
    df_summary.to_csv(out_csv, index=False, float_format="%.6f")

    # Save markdown version
    with open(out_md, "w") as f:
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


def run_variance_check(input_file: str) -> None:
    """Loads the summary file and calculates emission variance stats for each model.

    Args:
        input_file: Name of the summary file to analyze
    """
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
        emission_stats(model_df, model_name)


if __name__ == "__main__":
    run_summary(model_filter="_qwen3")
