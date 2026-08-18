"""
app/logging_config.py
----------------------
One function, one job: configure logging consistently across the FastAPI
server, the Streamlit app, and the pytest suite. Logs go to console AND to
a rotating file under logs/app.log so you can inspect failures after the
process has exited (useful in Docker).
"""
import logging
import logging.handlers
import sys

from app.config import LOG_DIR, LOG_LEVEL


def setup_logging(name: str = "biassistant") -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this is called more than once
    # (e.g. imported by both api and streamlit processes).
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "app.log", maxBytes=2_000_000, backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
