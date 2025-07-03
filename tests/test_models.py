from app.model_loader import load_model

def test_load_sklearn_model():
    model = load_model("sklearn")
    assert model is not None

def test_load_invalid_model():
    try:
        load_model("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_load_pytorch_model():
    try:
        model = load_model("pytorch")
        assert model is not None
    except ModuleNotFoundError:
        # pytorch might not be installed yet, so we skip
        pass

def test_load_tensorflow_model():
    try:
        model = load_model("tensorflow")
        assert model is not None
    except ModuleNotFoundError:
        # tensorflow might not be installed yet, so we skip
        pass
