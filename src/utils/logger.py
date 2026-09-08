import logging
import os
from constants.settings import settings

def setup_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger instance.
    """

    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    logger.setLevel(settings.LOG_LEVEL)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Application log
    file_handler = logging.FileHandler(
        "logs/app.log",
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Console log
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger