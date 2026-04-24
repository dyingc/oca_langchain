import os
import logging
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv

load_dotenv()


class _ConsoleFilter(logging.Filter):
    """Hide file-only verbose records from the console handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.ERROR:
            return True
        return getattr(record, "console", True)


def _ensure_parent_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger with:
    - File logging to LOG_FILE_PATH (from .env / environment)
    - Timed rotation every 5 days, archived files suffixed with timestamp
    - Level controlled by LOG_LEVEL (DEBUG/INFO)
    - Formatter with timestamp
    - DEBUG: also log to stdout
    """
    logger = logging.getLogger(name)

    # Force reload config to ensure we get the latest env vars
    log_file = os.getenv("LOG_FILE_PATH", "oca_llm.log")

    if logger.handlers:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()

    print(f"Configuring logger '{name}' writing to '{log_file}'")
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = logging.DEBUG if log_level_str == "DEBUG" else logging.INFO

    logger.setLevel(level)
    logger.propagate = False

    _ensure_parent_dir(log_file)

    fmt = logging.Formatter("%(asctime)s - %(message)s")

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="D",
        interval=5,
        utc=True,
        encoding="utf-8"
    )
    # archived filename: base + .YYYYmmdd_HHMMSS
    file_handler.suffix = "%Y%m%d_%H%M%S"
    file_handler.setFormatter(fmt)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    if level == logging.DEBUG:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.DEBUG)
        sh.addFilter(_ConsoleFilter())
        logger.addHandler(sh)

    return logger
