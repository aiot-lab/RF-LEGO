"""Base dataset abstractions for RF-LEGO modules."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from rflego.utils import logger


class BaseDataset(Dataset, ABC):
    """Base PyTorch dataset supporting pickle directories or in-memory sample arrays.

    Subclasses can either load pre-generated .pkl sample files from data_dir
    or wrap precomputed NumPy arrays produced by the synthetic batch builders.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        data: dict[str, np.ndarray] | None = None,
        n: int | None = None,
    ) -> None:
        super().__init__()
        self.data_dir: Path | None = Path(data_dir) if data_dir is not None else None
        self.file_paths: list[Path] = []
        self.data: dict[str, np.ndarray] | None = data
        self._n: int = 0

        if self.data_dir is not None:
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {self.data_dir}")
            self.file_paths = sorted(self.data_dir.glob("*.pkl"))
            if not self.file_paths:
                logger.warning(f"No .pkl files found in {self.data_dir}")
            self._n = len(self.file_paths)
        elif data is not None and n is not None:
            self._n = int(n)
        else:
            raise ValueError("Provide either 'data_dir' or in-memory ('data', 'n').")

    def _load_pickle(self, idx: int) -> dict[str, Any]:
        """Load a single pickled sample dictionary by index."""
        with self.file_paths[idx].open("rb") as f:
            return pickle.load(f)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self._n

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single sample dictionary."""
        raise NotImplementedError
