"""Structured logging setup — N40."""
import logging
import sys


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logger with structured format."""
    fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
    if json_format:
        # Try loguru first, fall back to standard
        try:
            from loguru import logger
            logger.remove()
            logger.add(sys.stdout, format="{time} {level} {name} {message}",
                      level=level, serialize=True)
            return
        except ImportError:
            pass
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
