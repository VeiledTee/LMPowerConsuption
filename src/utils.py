import logging
import os
import re
from pathlib import Path
from typing import Dict, Tuple

from config import CONFIG


def normalize(text: str) -> str:
    """
    Normalize text by collapsing whitespace, stripping, and lowercasing.

    Args:
        text (str): Input text string.

    Returns:
        str: Normalized string.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def strip_links(text: str) -> str:
    """
    Remove HTML anchor tags from a text string.

    Args:
        text (str): Input HTML-formatted string.

    Returns:
        str: String with anchor tags removed.
    """
    return re.sub(r"</?a[^>]*>", "", text).strip()


def convert_seconds(total_seconds: float) -> Tuple[int, int, int]:
    """
    Convert a time duration from seconds to hours, minutes, and seconds.

    Args:
        total_seconds (float): Duration in seconds.

    Returns:
        Tuple[int, int, int]: Hours, minutes, seconds.
    """
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return hours, minutes, seconds


def tail_row(path: Path) -> Dict[str, float]:
    """
    Extracts duration, emissions, and energy from the last row of a CSV file.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        Dict[str, float]: Dictionary with keys "duration", "emissions", "energy_consumed".
    """
    with path.open() as f:
        last = f.readlines()[-1].strip().split(",")
    return {
        "duration": float(last[4]),
        "emissions": float(last[5]),
        "energy_consumed": float(last[13]),
    }


def setup_logging() -> logging.Logger:
    """
    Set up logging to file and console using configuration in CONFIG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_file_path = CONFIG.energy_dir / "experiment.log"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logging.basicConfig(
        level=CONFIG.log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("energy_eval")


def ensure_config_dirs() -> None:
    """
    Ensure that all required directories defined in CONFIG exist.
    """
    dirs_to_check = [
        CONFIG.wiki_dir,
        CONFIG.corpus_cache.parent,
        CONFIG.tfidf_cache.parent,
        CONFIG.index_cache.parent,
        CONFIG.energy_dir,
        CONFIG.result_dir,
    ]
    for d in dirs_to_check:
        d.mkdir(parents=True, exist_ok=True)


def count_bools(output: str) -> str:
    """
    Heuristically determines a boolean answer ("True" or "False") from a free-form model output
    by counting occurrences of "true"/"yes" vs "false"/"no" (case-insensitive).

    Args:
        output (str): The model-generated response to a yes/no question.

    Returns:
        str: "True" if the output suggests a positive answer, "False" otherwise. Defaults to "True" on tie.
    """
    keywords = {"true": 0, "false": 0, "yes": 0, "no": 0}
    for word in output.lower().split():
        if word in keywords:
            keywords[word] += 1

    if keywords["true"] + keywords["yes"] > keywords["false"] + keywords["no"]:
        return "True"
    elif keywords["true"] + keywords["yes"] < keywords["false"] + keywords["no"]:
        return "False"

    return "True"  # Default majority class in BoolQ
