from config import CONFIG
from experiment import run
import logging

logger = logging.getLogger(__name__)


def safe_run(tag):
    logger.info(f"Starting run: {tag}")
    try:
        run()
    except Exception as e:
        logger.exception(f"Run {tag} failed: {e}")


# First config
CONFIG.dataset_file = "boolq_mini_128.jsonl"
safe_run("deepseek-mini")

# Second config
CONFIG.dataset_file = "boolq_full.jsonl"
safe_run("deepseek-full")
