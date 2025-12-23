import logging
import sys
import time
from pathlib import Path
from functools import wraps
from typing import Optional, Callable, Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)


def setup_logger(log_path: str, level: int = logging.INFO) -> None:
    """
    Setup global Root Logger.

    This function should be called once at the beginning of the program execution.

    Args:
        log_path: Path to the log directory or specific log file.
                  - If directory: creates 'YYYY-MM-DD_HH-MM-SS.log' inside.
                  - If file (ends with .txt/.log): appends to file tail.
        level: Logging level, default is logging.INFO.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")

    # Hijack existing StreamHandlers created by other packages
    existing_stream_handlers = [
        h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    for handler in existing_stream_handlers:
        root_logger.removeHandler(handler)

    _streamHandler = logging.StreamHandler(sys.stdout)
    _streamHandler.setFormatter(formatter)
    root_logger.addHandler(_streamHandler)

    # Resolve log file path
    path_obj = Path(log_path)
    target_file: Path

    if path_obj.suffix.lower() in [".txt", ".log"]:
        target_file = path_obj
        if not target_file.parent.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        if not path_obj.exists():
            path_obj.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        target_file = path_obj / f"{timestamp}.log"

    # Add FileHandler
    _fileHandler = logging.FileHandler(str(target_file), mode="a", encoding="utf-8")
    _fileHandler.setFormatter(formatter)
    root_logger.addHandler(_fileHandler)

    root_logger.info(f"Log initialized. Saving to: {target_file}")


def log_exception(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to capture exceptions raised in the decorated function,
    log the full traceback to the configured logger, and re-raise the exception.

    Usage:
        @log_exception
        def my_function():
            ...

    Args:
        func: The function to decorate.

    Returns:
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.getLogger().exception(
                f"Exception occurred in function '{func.__name__}': {str(e)}"
            )
            raise e

    return wrapper
