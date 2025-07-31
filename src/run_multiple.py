from config import CONFIG
from experiment import run
import logging
from analysis import main

logger = logging.getLogger(__name__)


def safe_run(tag, file_suffix: None | str = ""):
    logger.info(f"Starting run: {tag} - {file_suffix}")
    try:
        run(file_suffix)
    except Exception as e:
        logger.exception(f"Run {tag} failed: {e}")


# # First config
# CONFIG.think = True
# safe_run("deepseek-think")
# # main("_deepseek")

CONFIG.dataset_name = "hotpotqa/hotpot_qa"
CONFIG.dataset_file = "hotpot_mini_dev_128.jsonl"
CONFIG.think = False

# # config 2
# CONFIG.model_types = {
#     "gemma3:4b": "ollama",
# }
# CONFIG.modes = {
#     "gemma3:4b": {"q": False, "q+r": True},
# }
# for i in range(5):
#     safe_run("gemma3_4_dev_testing", str(i))
#
# # config 3
# CONFIG.model_types = {
#     "gemma3:12b": "ollama",
# }
# CONFIG.modes = {
#     "gemma3:12b": {"q": False, "q+r": True},
# }
# for i in range(5):
#     safe_run("gemma3_12_dev_testing", str(i))

# # config 4
# CONFIG.model_types = {
#     "gemma3:27b": "ollama",
# }
# CONFIG.modes = {
#     "gemma3:27b": {"q": False},
# }
# for i in range(5):
#     safe_run("gemma3_27_dev_testing", str(i))

# config 5
CONFIG.think = True
CONFIG.model_types = {
    "deepseek-r1:8b": "ollama",
}
CONFIG.modes = {
    "deepseek-r1:8b": {"q": False},
}
for i in range(5):
    safe_run("deepseek_8_dev_testing", str(i))

# # config 6
# CONFIG.model_types = {
#     "deepseek-r1:14b": "ollama",
# }
# CONFIG.modes = {
#     "deepseek-r1:14b": {"q": False, "q+r": True},
# }
# for i in range(5):
#     safe_run("deepseek_14_dev_testing", str(i))
#
# # config 7
# CONFIG.model_types = {
#     "deepseek-r1:32b": "ollama",
# }
# CONFIG.modes = {
#     "deepseek-r1:32b": {"q": False},
# }
# for i in range(5):
#     safe_run("deepseek_32_dev_testing", str(i))
