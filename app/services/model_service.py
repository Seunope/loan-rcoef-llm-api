import joblib
import numpy as np

# Load models
linear_model = joblib.load("models/linear_reg.pkl")
random_forest_model = joblib.load("models/random_forest.pkl")

def predict(model_type: str, input_data: list[float]):
    """Perform prediction based on model type"""
    data = np.array(input_data).reshape(1, -1)

    if model_type == "linear":
        return linear_model.predict(data).tolist()
    elif model_type == "random_forest":
        return random_forest_model.predict(data).tolist()
    else:
        return {"error": "Invalid model_type"}
