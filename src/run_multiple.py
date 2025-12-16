import logging
import os

from src.config import CONFIG
from src.experiment import run, send_email

logger = logging.getLogger(__name__)


def safe_run(tag, file_suffix: None | str = ""):
    logger.info(f"Starting run: {tag} - {file_suffix}")
    try:
        run(file_suffix)
    except Exception as e:
        logger.exception(f"Run {tag} failed: {e}")


# # First config
# CONFIG.think = False
# CONFIG.gold = True
# safe_run("gemma3-hotpot")
# send_email(
#     from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
#     to_addr=CONFIG.to_email,
#     subject="Completed gemma3-hotpot with no issues.",
#     body="No errors",
# )

CONFIG.model_types = {
    "qwen3:0.6b": "ollama",
    "qwen3:1.7b": "ollama",
    "qwen3:4b": "ollama",
    "qwen3:8b": "ollama",
    "qwen3:14b": "ollama",
    "qwen3:32b": "ollama",
}
CONFIG.modes = {
    "qwen3:0.6b": {"q": False, "q+r": True},
    "qwen3:1.7b": {"q": False, "q+r": True},
    "qwen3:4b": {"q": False, "q+r": True},
    "qwen3:8b": {"q": False, "q+r": True},
    "qwen3:14b": {"q": False, "q+r": True},
    "qwen3:32b": {"q": False},
 }

# # 1st config
# CONFIG.think = False
# CONFIG.gold = True
# try:
#     safe_run("qwen3-hotpot-nothink")
#     send_email(
#         from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
#         to_addr=CONFIG.to_email,
#         subject="Completed qwen3-hotpot no think with no issues.",
#         body="No errors",
#     )
# except Exception as error:
#     send_email(
#         from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
#         to_addr=CONFIG.to_email,
#         subject="Script Error",
#         body=f"The qwen3-hotpot-nothink script crashed with:\n\n{error}",
#     )

# 2nd config
CONFIG.think = True
CONFIG.gold = True
try:
    safe_run("qwen3-hotpot-think")
    send_email(
        from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
        to_addr=CONFIG.to_email,
        subject="Completed qwen3-hotpot think with no issues.",
        body="No errors",
    )
except Exception as error:
    send_email(
        from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
        to_addr=CONFIG.to_email,
        subject="Script Error",
        body=f"The qwen3-hotpot-think script crashed with:\n\n{error}",
    )

# CONFIG.model_types = {
#     "deepseek-r1:1.5b": "ollama",
#     "deepseek-r1:7b": "ollama",
#     "deepseek-r1:14b": "ollama",
#     "deepseek-r1:32b": "ollama"
# }
# CONFIG.modes = {
#     "deepseek-r1:1.5b": {"q": False, "q+r": True},
#     "deepseek-r1:7b": {"q": False, "q+r": True},
#     "deepseek-r1:14b": {"q": False, "q+r": True},
#     "deepseek-r1:32b": {"q": False}
#  }
#
# # First config
# CONFIG.think = True
# CONFIG.gold = True
# safe_run("deepseek-hotpot")
# send_email(
#     from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
#     to_addr=CONFIG.to_email,
#     subject="Completed deepseek-hotpot with no issues.",
#     body="No errors",
# )
