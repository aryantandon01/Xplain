def load_model(model_type: str):
    if model_type == "sklearn":
        from app.models import sklearn_model

        return sklearn_model.load_model()
    elif model_type == "pytorch":
        from app.models import pytorch_model

        return pytorch_model.load_model()
    elif model_type == "tensorflow":
        from app.models import tensorflow_model

        return tensorflow_model.load_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
