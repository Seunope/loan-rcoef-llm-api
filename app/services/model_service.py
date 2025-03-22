import joblib
import numpy as np
from pydantic import BaseModel

# Load models
linear_model = joblib.load("models/ml/model/linear_reg.pkl")
random_forest_model = joblib.load("models/ml/model/random_forest.pkl")
neural_network_model = joblib.load("models/ml/model/neural_network.pkl")

# Load pre-trained encoders and scaler
encoders = joblib.load("models/ml/scale/encoders.pkl")  # Dictionary of LabelEncoders
scaler = joblib.load("models/ml/scale/scaler.pkl")      # StandardScaler instance

# Define feature order
categorical_cols = ["gender", "maritalStatus", "state"]
numerical_cols = ["age", "loanAmount", "tenorInDays"]
feature_order = categorical_cols + numerical_cols  # Ensures model receives input in the correct order

# Define request schema
class ModelInput(BaseModel):
    model_type: str
    features: dict  


def predict(data: ModelInput):

    """Predict based on input data after encoding and scaling."""
    features = data.features
    ml_model = data.model_type

    # Ensure all required features are present
    missing_cols = [col for col in feature_order if col not in features]
    if missing_cols:
        return {"error": f"Missing required columns: {missing_cols}"}
    
    
    # Encode categorical variables using stored encoders
    encoded_data = []
    for col in categorical_cols:
        if col in encoders:
            encoder = encoders[col]
            try:
                encoded_data.append(int(encoder.transform([features[col]])[0]))  # Ensure integer encoding
            except ValueError:
                return {"error": f"Invalid value '{features[col]}' for '{col}'"}
        else:
            return {"error": f"Missing encoder for '{col}'"}

    # Scale numerical variables using the stored scaler
    numerical_data = np.array([[features[col] for col in numerical_cols]])  # Extract numerical features
    scaled_numerical_data = scaler.transform(numerical_data)[0].tolist()  # Scale them

    # Combine encoded categorical + scaled numerical features
    final_input = np.array(encoded_data + scaled_numerical_data).reshape(1, -1)


    if ml_model == "linear":
        return linear_model.predict(final_input).tolist()
    elif ml_model == "random_forest":
        return random_forest_model.predict(final_input).tolist()
    elif ml_model == "neural_network":
        return random_forest_model.predict(final_input).tolist()
    else:
        return {"error": "Invalid model_type"}
