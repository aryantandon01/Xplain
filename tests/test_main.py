from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_explain_endpoint():
    response = client.post("/explain", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert "shap_values" in response.json()
