import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

NUM_ATTRIBS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
CAT_ATTRIBS = ["ocean_proximity"]
OCEAN_PROXIMITY_OPTIONS = ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]

# Median-ish values from the training data, used to pre-fill the form
FORM_DEFAULTS = {
    "longitude": -119.57,
    "latitude": 35.63,
    "housing_median_age": 28,
    "total_rooms": 2127,
    "total_bedrooms": 435,
    "population": 1166,
    "households": 409,
    "median_income": 3.53,
    "ocean_proximity": "<1H OCEAN",
}


def render_page(**overrides):
    context = {
        "options": OCEAN_PROXIMITY_OPTIONS,
        "form_values": FORM_DEFAULTS,
        "result": None,
        "error": None,
    }
    context.update(overrides)
    return render_template("index.html", **context)


@app.route("/", methods=["GET"])
def index():
    return render_page()


@app.route("/predict", methods=["POST"])
def predict():
    submitted = {field: request.form.get(field, "") for field in NUM_ATTRIBS}
    submitted["ocean_proximity"] = request.form.get("ocean_proximity", "")

    try:
        row = {}
        for field in NUM_ATTRIBS:
            raw = submitted[field]
            if raw == "":
                raise ValueError(f"'{field.replace('_', ' ')}' is required.")
            row[field] = float(raw)

        if submitted["ocean_proximity"] not in OCEAN_PROXIMITY_OPTIONS:
            raise ValueError("Choose a valid ocean proximity option.")
        row["ocean_proximity"] = submitted["ocean_proximity"]

        input_df = pd.DataFrame([row], columns=NUM_ATTRIBS + CAT_ATTRIBS)
        transformed = pipeline.transform(input_df)
        prediction = float(model.predict(transformed)[0])

        return render_page(form_values=submitted, result=round(prediction, 2))

    except Exception as exc:
        return render_page(form_values=submitted, error=str(exc))


if __name__ == "__main__":
    app.run(debug=True)
