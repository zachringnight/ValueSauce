"""Base model interface for all prediction models."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    model_name: str
    model_version: str
    git_commit_hash: Optional[str] = None
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    validation_window: Optional[str] = None
    hyperparams: Optional[dict] = None
    metrics: Optional[dict] = None
    artifact_uri: Optional[str] = None


class BaseModel(ABC):
    """Abstract base class for all models in the pipeline."""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.is_fitted = False
        self.metadata: Optional[ModelMetadata] = None
        self.logger = logging.getLogger(f"{__name__}.{name}")

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate point predictions."""
        pass

    def get_metadata(self) -> ModelMetadata:
        if self.metadata is None:
            self.metadata = ModelMetadata(
                model_name=self.name,
                model_version=self.version,
            )
        return self.metadata

    def save(self, path: str) -> None:
        """Save model artifacts to disk."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model artifacts from disk."""
        import pickle
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
