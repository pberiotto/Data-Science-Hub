from abc import ABC, abstractmethod
from pathlib import Path

import joblib


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path):
        return joblib.load(Path(path))
