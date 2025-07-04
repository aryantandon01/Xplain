# Xplain - Explainable AI Microservice

Xplain is a lightweight, model-agnostic Explainable AI (XAI) microservice built with FastAPI.
It provides real-time feature-level explanations for tabular ML models using SHAP and LIME out of the box, with extensible architecture to support more explainers and models.

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
## Project Structure

```bash
Xplain
├── app/
│   ├── explainers/              # SHAP & LIME wrappers
│   │   ├── __init__.py
│   │   ├── lime_explainer.py
│   │   └── shap_explainer.py
│   ├── models/                  # Example models
│   │   ├── sklearn_model.py
│   │   ├── tensorflow_model.py
│   │   └── pytorch_model.py
│   ├── main.py                  # FastAPI app
│   ├── model_loader.py
│   ├── schemas.py
│   └── training_data.npy        # Data for LIME/SHAP background
├── tests/                       # Tests
│   ├── test_main.py
│   └── test_models.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```
---

## Quickstart (3 Methods to run this)

### Run locally (development)

```bash
git clone https://github.com/aryantandon01/Xplain.git
cd Xplain
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
    [
      "f1",
      0.06521656829575624
    ],
    [
      "f2",
      -0.025804609295375428
    ],
    [
      "f3",
      -0.03941195900038079
    ]
  ]
}
```

---

## Extending

### Add a new explainer

1. Create a new file in `app/explainers/`, e.g., `your_explainer.py`
2. Implement a function like `your_explain(...)` that takes `(model, input_features)`
3. Update `app/explainers/__init__.py` to import and register it inside the `explain` function
4. Use it via query param: `?explainer=your_explainer` on the `/explain` endpoint

---

### Add a new model type

1. Add your wrapper in `app/models/`, e.g., `my_model.py`
2. Implement `load_model()` to return the trained model
3. Update `app/model_loader.py` to load your new model type when `model_type="my_model"`
4. Use it via query param: `?model_type=my_model` on both `/predict` and `/explain` endpoints

---

## License

MIT License – you are free to use, modify, and share this project.

---

## Contributing

Currently in early development.
If you'd like to contribute, open an issue or submit a pull request.

---
