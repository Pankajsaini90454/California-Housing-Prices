# California Housing — Appraisal Tool

A Flask front end for the `RandomForestRegressor` + preprocessing `Pipeline` trained in your `main.py`. Enter a parcel's location and census-block stats, get back a predicted `median_house_value`.

## Files

```
housing_app/
├── app.py              # Flask routes + inference
├── model.pkl           # trained RandomForestRegressor (copied from your project)
├── pipeline.pkl         # fitted ColumnTransformer (imputer/scaler/one-hot)
├── templates/index.html
├── static/style.css
├── requirements.txt
└── Procfile             # for gunicorn-based hosts (Render, Railway, Heroku)
```

## Run locally

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```

Visit `http://127.0.0.1:5000`.

## How it works

- `app.py` loads `model.pkl` and `pipeline.pkl` once at startup with `joblib.load`.
- The form's 8 numeric fields + `ocean_proximity` are assembled into a one-row `DataFrame` with the **same column names and order** the pipeline was fit on.
- `pipeline.transform(...)` → `model.predict(...)` → result rendered back into the page. No page reload framework, no JS build step — just a POST to `/predict`.
- Bad or missing input is caught and shown as an inline error instead of a 500 page.

## Before you deploy — two things worth fixing

1. **Model file size.** `model.pkl` is ~144 MB (a `RandomForestRegressor` with default `n_estimators=100` on this dataset). Some free hosting tiers (Heroku, some Render free plans) cap slug/image size well under that. Two ways to shrink it, either works:
   - Retrain with fewer/shallower trees: `RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)` — cuts size a lot with a small accuracy trade-off.
   - Or keep accuracy and switch to a host with no tight size cap (Render paid tier, Railway, a small VPS, AWS/GCP).

2. **scikit-learn version match.** Your pickles were made with scikit-learn 1.9.0. Pin the exact same version in `requirements.txt` (already done here) — loading a pickle with a different sklearn version prints `InconsistentVersionWarning` and can silently change results.

## Deploying (Render, as an example)

1. Push this folder to a GitHub repo.
2. On Render: New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Deploy. Render reads the `Procfile` too, so the start command is set automatically.

Railway and Fly.io follow the same shape (`Procfile` + `requirements.txt`); for a raw VPS, run `gunicorn -b 0.0.0.0:8000 app:app` behind nginx.
