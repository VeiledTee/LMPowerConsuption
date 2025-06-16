import logging
import re
from pathlib import Path
from config import CONFIG


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

def strip_links(text: str) -> str:
    return re.sub(r"</?a[^>]*>", "", text).strip()

def convert_seconds(total_seconds):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return hours, minutes, seconds

def tail_row(path: Path):
    with path.open() as f:
        last = f.readlines()[-1].strip().split(",")
    return {
        "duration": float(last[4]),
        "emissions": float(last[5]),
        "energy_consumed": float(last[13]),
    }

def setup_logging():
    logging.basicConfig(
        level=CONFIG.log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG.energy_dir / 'experiment.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("energy_eval")


def ensure_config_dirs():
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