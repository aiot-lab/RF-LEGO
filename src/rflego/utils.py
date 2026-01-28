"""Utility functions and centralized logging for RF-LEGO.

This module provides:
- Centralized logging configuration using loguru
- Common utility functions shared across the library
- Device detection and management
"""

import sys
from pathlib import Path

import torch
from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: Path | str | None = None,
    format_str: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
) -> None:
    """Configure the centralized logger for RF-LEGO.

    Sets up loguru with custom formatting and optional file output.

    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path for file logging.
        format_str: Custom format string for log messages.

    Example:
        >>> from rflego.utils import setup_logger
        >>> setup_logger(level="DEBUG", log_file="./logs/training.log")
    """
    # Remove default handler
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stderr,
        format=format_str,
        level=level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            format=format_str.replace("<green>", "")
            .replace("</green>", "")
            .replace("<level>", "")
            .replace("</level>", "")
            .replace("<cyan>", "")
            .replace("</cyan>", ""),
            level=level,
            rotation="10 MB",
            retention="7 days",
        )

    logger.debug(f"Logger initialized with level={level}")


def get_device(device: str | None = None) -> torch.device:
    """Get the best available device for computation.

    Automatically detects CUDA or MPS availability if device is not specified.

    Args:
        device: Explicit device string ('cpu', 'cuda', 'mps', 'cuda:0', etc.).
                If None, auto-detects the best available device.

    Returns:
        torch.device: The selected device.

    Example:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cuda:1")  # Specific GPU
    """
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a model.

    Args:
        model: PyTorch model to analyze.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Total number of parameters.

    Example:
        >>> model = DetectorModel(config)
        >>> total_params = count_parameters(model)
        >>> print(f"Trainable parameters: {total_params:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(num_params: int) -> str:
    """Format parameter count with human-readable suffixes.

    Args:
        num_params: Number of parameters.

    Returns:
        Formatted string (e.g., "1.5M", "256K").
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Sets seeds for Python random, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed: Random seed value.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.debug(f"Random seed set to {seed}")


def ensure_dir(path: Path | str) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Export logger for direct use
__all__ = [
    "logger",
    "setup_logger",
    "get_device",
    "count_parameters",
    "format_params",
    "set_seed",
    "ensure_dir",
]
