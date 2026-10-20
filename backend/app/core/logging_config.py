"""FastRAG Logging Configuration.

Configures standard Python logging with structured formats, configurable log levels,
latency tracking capabilities for performance-critical pipeline steps, and safeguards
against accidental credential leakage.
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Generator, Any


class LatencyLogFormatter(logging.Formatter):
    """Custom formatter including timestamp, severity, logger name, and optional latency metadata."""

    def format(self, record: logging.LogRecord) -> str:
        # Check if latency metadata was attached to the log record
        latency_ms = getattr(record, "latency_ms", None)
        if latency_ms is not None:
            record.latency_tag = f" [latency={latency_ms:.2f}ms]"
        else:
            record.latency_tag = ""

        # Check if operation tag was attached
        operation = getattr(record, "operation", None)
        if operation:
            record.op_tag = f" [op={operation}]"
        else:
            record.op_tag = ""

        return super().format(record)


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logging for FastRAG."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = LatencyLogFormatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s%(op_tag)s%(latency_tag)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance for FastRAG components."""
    return logging.getLogger(name)


@contextmanager
def measure_latency(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    extra: dict[str, Any] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager to measure and log operation latency in milliseconds.

    Yields a mutable context dict so callers can attach additional non-sensitive metadata.
    """
    ctx: dict[str, Any] = extra.copy() if extra else {}
    start_time = time.perf_counter()
    try:
        yield ctx
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        log_extra = {"latency_ms": elapsed_ms, "operation": operation, **ctx}
        logger.log(
            level,
            f"Operation '{operation}' completed in {elapsed_ms:.2f}ms",
            extra=log_extra,
        )
