import joblib
import numpy as np


def predict_ecg(signal_features, model_path="knn_model.pkl"):

    model = joblib.load(model_path)

    prediction = model.predict([signal_features])

    return prediction[0]


if __name__ == "__main__":

    # Example feature vector
    features = np.random.rand(20)

    disease = predict_ecg(features)

    print("Predicted disease:", disease)
