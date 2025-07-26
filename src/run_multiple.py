from config import CONFIG
from experiment import run
import logging
from analysis import main

logger = logging.getLogger(__name__)


def safe_run(tag, file_suffix: None | str = ""):
    logger.info(f"Starting run: {tag}")
    try:
        run(file_suffix)
    except Exception as e:
        logger.exception(f"Run {tag} failed: {e}")


# # First config
# CONFIG.think = True
# safe_run("deepseek-think")
# # main("_deepseek")

# Second config
# CONFIG.model_types = {
#     "gemma3:4b": "ollama",
# }
# CONFIG.modes = {
#     "gemma3:4b": {"q+r": True},
# }
CONFIG.think = False
for i in range(5):
    safe_run("gemma3_dev_testing", str(i))

# main("hotpot_deepseek")
