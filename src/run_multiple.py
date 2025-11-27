import logging

from src.config import CONFIG
from src.experiment import run

logger = logging.getLogger(__name__)


def safe_run(tag, file_suffix: None | str = ""):
    logger.info(f"Starting run: {tag} - {file_suffix}")
    try:
        run(file_suffix)
    except Exception as e:
        logger.exception(f"Run {tag} failed: {e}")


# First config
CONFIG.think = False
CONFIG.gold = True
safe_run("qen3-no_think-gs")

# Second config
CONFIG.think = False
CONFIG.gold = False
safe_run("qen3-no_think-fp")
