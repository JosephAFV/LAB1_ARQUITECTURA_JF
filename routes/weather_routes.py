"""Adaptador HTTP que delega los casos de uso."""
from flask import Blueprint, current_app, flash, render_template, request
web = Blueprint("web", __name__)

def service():
    return current_app.extensions["weather_service"]

def page(prediction_result=None, status=200):
    return render_template("index.html", **service().state(),
                           prediction_result=prediction_result,
                           form_values=request.form), status

@web.get("/")
def index():
    if request.args.get("nuevo") == "1":
        return render_template("index.html", results=None, prediction_features=[],
                               prediction_schema=[], prediction_result=None, form_values={})
    return page()

@web.post("/train")
def train():
    uploaded = request.files.get("dataset")
    if uploaded is None or not uploaded.filename:
        flash("Selecciona un archivo CSV.")
        return page(status=400)
    if not uploaded.filename.lower().endswith(".csv"):
        flash("El archivo debe tener extensión .csv.")
        return page(status=400)
    try:
        service().train(uploaded.stream)
        return page()
    except ValueError as exc:
        flash(str(exc))
        return page(status=400)
    except Exception:
        current_app.logger.exception("Error durante el entrenamiento")
        flash("No se pudo procesar el dataset. Revisa el archivo e inténtalo nuevamente.")
        return page(status=400)

@web.post("/predict")
def predict():
    try:
        return page(service().predict(request.form))
    except ValueError as exc:
        flash(str(exc))
        return page(status=400)
    except Exception:
        current_app.logger.exception("Error durante la predicción")
        flash("No se pudo realizar la predicción. Revisa los datos ingresados.")
        return page(status=400)
