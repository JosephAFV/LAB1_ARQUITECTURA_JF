"""Persistencia local del modelo y sus metadatos."""
from pathlib import Path
import joblib

class ModelRepository:
    def __init__(self, path):
        self.path = Path(path)

    def exists(self):
        return self.path.exists()

    def save(self, bundle):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, self.path)

    def load(self):
        if not self.exists():
            raise ValueError("Primero entrena un modelo subiendo un dataset.")
        return joblib.load(self.path)
