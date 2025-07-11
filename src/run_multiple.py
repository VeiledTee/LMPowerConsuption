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
CONFIG.think = True
safe_run("deepseek-big-think")

# Second config
CONFIG.think = False
safe_run("deepseek-smooth-brain")
