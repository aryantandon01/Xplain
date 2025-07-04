# Xplain - Explainable AI Microservice

Xplain is a lightweight, model-agnostic Explainable AI (XAI) microservice built with FastAPI.
It provides real-time feature-level explanations for tabular ML models using SHAP and LIME, with extensible architecture to support more explainers and models.

---

## Key Features

* REST API built with FastAPI for predictions and explanations
* Plug-and-play support for **SHAP** and **LIME** explainers
* Works with scikit-learn, PyTorch, and TensorFlow models
* Tracks inputs, outputs, and metadata using MLflow
* Docker and Docker Compose setup for easy deployment
* Modular design to add custom explainers or models

---

## Technologies Used

* FastAPI (API)
* MLflow (experiment tracking)
* SHAP & LIME (explainers)
* scikit-learn, PyTorch, TensorFlow (model backends)
* Docker, Docker Compose (containerization)
* Streamlit (optional dashboard)

---

## Quickstart

### Run locally (development)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t xplain-api .
docker run -p 8000:8000 xplain-api
```

### Run full stack (API + MLflow UI)

```bash
docker compose up --build
```

* FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
* MLflow UI: [http://localhost:5000](http://localhost:5000)

---

## API Overview

### `POST /predict`

Request:

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

### `POST /explain`

Query parameter: `explainer=shap` or `explainer=lime` (default: shap)

Request:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Response:

```json
{
  "shap_values": [
    ["f1 <= 5.10", -0.02],
    ["f2 > 3.40", 0.03]
  ]
}
```

---

## Extending

To add a new explainer:

1. Create a file in `app/explainers/`, e.g., `your_explainer.py`
2. Update `app/explainers/__init__.py` to register it
3. It will be available via `?explainer=your_explainer` in the `/explain` endpoint

To add new models:

* Place the wrapper under `app/models/`
* Update `app/model_loader.py` accordingly

---

## Project Structure

```
.
├── app/
│   ├── explainers/
│   ├── models/
│   ├── main.py
│   └── ...
├── dashboard/ (optional Streamlit dashboard)
├── notebooks/ (experiments & demos)
├── tests/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── ...
```

---

## License

MIT License – you are free to use, modify, and share this project.

---

## Contributing

Currently in early development.
If you'd like to contribute, open an issue or submit a pull request.

---
