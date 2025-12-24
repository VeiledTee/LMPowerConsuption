import logging
import os

from config import CONFIG
from experiment import run, send_email

logger = logging.getLogger(__name__)


def safe_run(tag, file_suffix: None | str = ""):
    logger.info(f"Starting run: {tag} - {file_suffix}")
    try:
        run(file_suffix)
    except Exception as e:
        logger.exception(f"Run {tag} failed: {e}")


# First config
CONFIG.think = True
CONFIG.gold = True
try:
    safe_run("deepseek-8B-GOLD-NQ")
    send_email(
        from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
        to_addr=CONFIG.to_email,
        subject="Completed gemma3-NQ with no issues.",
        body="No errors",
    )
except Exception as error:
    send_email(
        from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
        to_addr=CONFIG.to_email,
        subject="Script Error",
        body=f"The deepseek-8B-GOLD-NQ script crashed with:\n\n{error}",
    )

CONFIG.gold = False
try:
    safe_run("deepseek-8B-FP-NQ")
    send_email(
        from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
        to_addr=CONFIG.to_email,
        subject="Completed gemma3-NQ with no issues.",
        body="No errors",
    )
except Exception as error:
    send_email(
        from_addr=os.getenv("EMAIL_FROM", CONFIG.from_email),
        to_addr=CONFIG.to_email,
        subject="Script Error",
        body=f"The deepseek-8B-FP-NQ script crashed with:\n\n{error}",
    )