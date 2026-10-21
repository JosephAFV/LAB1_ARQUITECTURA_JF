"""Casos de uso independientes de Flask; conserva los cinco filtros."""
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from filtros import (LoadDatasetFilter, ValidateDatasetFilter, PrepareDataFilter,
                     TrainModelFilter, EvaluateModelFilter)

class WeatherService:
    def __init__(self, repository):
        self.repository = repository

    def train(self, source):
        raw = LoadDatasetFilter().run(source)
        data = ValidateDatasetFilter("rain").run(raw)
        X, y, info = PrepareDataFilter("rain").run(data)
        classes = y.nunique()
        test_rows = math.ceil(len(y) * .20)
        if y.value_counts().min() < 2 or test_rows < classes or len(y)-test_rows < classes:
            raise ValueError("Datos insuficientes para separar entrenamiento y prueba por clase.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.20, random_state=42, stratify=y)
        model = TrainModelFilter().run(X_train, y_train, info)
        metrics = EvaluateModelFilter().run(model, X_test, y_test)
        schema = [{"name": name, "numeric": name in info["numeric_columns"]} for name in X.columns]
        results = {"rows": len(data), "features": len(X.columns), "target": "rain",
            "train_rows": len(X_train), "test_rows": len(X_test),
            "class_distribution": {str(k): int(v) for k, v in y.value_counts().items()},
            "metrics": metrics, "model_path": str(self.repository.path)}
        self.repository.save({"model": model, "features": list(X.columns),
                              "target": "rain", "schema": schema, "results": results})
        return results

    def state(self):
        if not self.repository.exists():
            return {"results": None, "prediction_features": [], "prediction_schema": []}
        saved = self.repository.load()
        schema = saved.get("schema")
        if schema is None:
            preprocessor = saved["model"].named_steps["preprocessor"]
            numeric = next((list(cols) for name, _, cols in preprocessor.transformers_ if name == "numeric"), [])
            schema = [{"name": n, "numeric": n in numeric} for n in saved["features"]]
        return {"results": saved.get("results"), "prediction_features": saved["features"],
                "prediction_schema": schema}

    def predict(self, form):
        saved = self.repository.load()
        values = {}
        for field in self.state()["prediction_schema"]:
            name = field["name"]
            raw = str(form.get(name, "")).strip()
            if not raw:
                raise ValueError(f"El campo '{name}' es obligatorio.")
            if field["numeric"]:
                try:
                    value = float(raw)
                except ValueError:
                    raise ValueError(f"El campo '{name}' debe ser numérico.") from None
                if not math.isfinite(value):
                    raise ValueError(f"El campo '{name}' debe ser un número finito.")
                values[name] = value
            else:
                values[name] = raw
        sample = pd.DataFrame([values], columns=saved["features"])
        model = saved["model"]
        return {"label": str(model.predict(sample)[0]),
                "probability": round(float(max(model.predict_proba(sample)[0])) * 100, 2)}
