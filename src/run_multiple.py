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
safe_run("gemma3-gs")

# Second config
CONFIG.think = False
CONFIG.gold = False
safe_run("gemma3-fp")

CONFIG.modes = {
    "deepseek-r1:1.5b": {"q": False, "q+r": True},
    "deepseek-r1:7b": {"q": False, "q+r": True},
    "deepseek-r1:8b": {"q": False, "q+r": True},
    "deepseek-r1:14b": {"q": False, "q+r": True},
    "deepseek-r1:32b": {"q": False}
 }
CONFIG.model_types = {
    "deepseek-r1:1.5b": "ollama",
    "deepseek-r1:7b": "ollama",
    "deepseek-r1:8b": "ollama",
    "deepseek-r1:14b": "ollama",
    "deepseek-r1:32b": "ollama"
}

# First config
CONFIG.think = True
CONFIG.gold = True
safe_run("deepseek-gs")

# Second config
CONFIG.think = True
CONFIG.gold = False
safe_run("deepseek-fp")
