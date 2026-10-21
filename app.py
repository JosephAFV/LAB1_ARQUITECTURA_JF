"""Punto de entrada y composición de dependencias."""
import os
from pathlib import Path
from flask import Flask
from repositories.model_repository import ModelRepository
from services.weather_service import WeatherService
from routes.weather_routes import web


def create_app(test_config=None):
    app = Flask(__name__, static_folder="public", static_url_path="/public")
    app.config.from_mapping(SECRET_KEY=os.environ.get("SECRET_KEY", "weather-ml-laboratorio"),
        MODEL_PATH=Path(__file__).parent / "models" / "weather_model.joblib",
        MAX_CONTENT_LENGTH=10 * 1024 * 1024)
    if test_config:
        app.config.update(test_config)
    app.extensions["weather_service"] = WeatherService(ModelRepository(app.config["MODEL_PATH"]))
    app.register_blueprint(web)
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)
