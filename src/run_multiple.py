from config import CONFIG
from experiment import run
import logging
from analysis import main

logger = logging.getLogger(__name__)


def safe_run(tag):
    logger.info(f"Starting run: {tag}")
    try:
        run()
    except Exception as e:
        logger.exception(f"Run {tag} failed: {e}")


# # First config
# CONFIG.think = True
# safe_run("deepseek-big-think")
main("_deepseek")

# Second config
CONFIG.model_types = {
    "gemma3:4b": "ollama",
    "gemma3:12b": "ollama",
    "gemma3:27b": "ollama",
}
CONFIG.modes = {
    "gemma3:4b": {"q": False, "q+r": True},
    "gemma3:12b": {"q": False, "q+r": True},
    "gemma3:27b": {"q": False},
}
CONFIG.think = False
safe_run("gemma3-big")

main("_gemma3")
