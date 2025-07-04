# Xplain - XAI Microservice

A lightweight, model-agnostic **Explainable AI (XAI) microservice** built with **FastAPI**, supporting SHAP & LIME explanations out of the box.  
Use it to generate feature-level explanations for tabular ML models in real time — locally, in Docker, or in production.

---

## Features
- FastAPI-based REST API for predictions & explanations
- Plug-and-play explainers: **SHAP**, **LIME**
- Tracks inputs & outputs using **MLflow**
- Easily extensible: add your own custom explainers
- Containerized with Docker & Docker Compose

---

## Quickstart

### Run locally (dev mode)
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t xplain-api .
docker run -p 8000:8000 xplain-api
```

### Run full stack (API + MLflow) with Docker Compose

```bash
docker compose up --build
```

* FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* MLflow UI: [http://localhost:5000](http://localhost:5000)

---

## API Endpoints

### `POST /predict`

Get model prediction for given input features.

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response:

```json
{
  "prediction": 0
}
```

---

### `POST /explain`

Get feature attribution for the prediction.

**Query param:** `explainer=shap` or `explainer=lime` (default: `shap`)

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response (SHAP or LIME output, simplified):

```json
{
  "shap_values": [
    ["f1 <= 5.10", -0.02],
    ["f2 > 3.40", 0.03],
    ...
  ]
}
```

---

## Extending

* Add a new explainer: create `app/explainers/your_explainer.py`
* Update `app/explainers/__init__.py` to register it
* Done! It’ll be available via `?explainer=your_explainer`

---

## License

MIT — feel free to use, share, and build on it.

---

## 🤝 Contributing

Coming soon. For now, open an issue or PR!
