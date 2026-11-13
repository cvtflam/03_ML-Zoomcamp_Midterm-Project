
from pathlib import Path
from functools import lru_cache
import pickle
import pandas as pd
from train import preprocess_data

# 1. Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.bin"

# 2. Load model and DictVectorizer
def load_model():

    with open(MODEL_PATH, "rb") as f_in:
        artifacts = pickle.load(f_in)

    model = artifacts["model"]
    dv = artifacts["dv"]

    return model, dv


# 3. Prediction

def predict(track_data):

    # Load trained model and vectorizer
    model, dv = load_model()

    # Convert incoming dictionary into DataFrame
    df = pd.DataFrame([track_data])

    # Apply the same feature engineering as training
    df_processed = preprocess_data(df)

    # Remove target column if accidentally supplied
    df_processed = df_processed.drop(
        columns=["popularity"],
        errors="ignore"
    )

    # Transform using the ORIGINAL fitted vectorizer
    X = dv.transform(
        df_processed.to_dict(orient="records")
    )

    # Generate prediction
    prediction = model.predict(X)

    return float(prediction[0])


if __name__ == "__main__":

    # Simple local test
    test_track = {
        "track_name": "Example Track",
        "track_id": "123456",
        "album_name": "Example Album",
        "release_year": 2023,
        "danceability": 0.8,
        "loudness": -5.0,
        "tempo": 188.0,
        "speechiness": 0.03,
        "acousticness": 0.1,
    }

    result = predict(test_track)

    print(f"Predicted popularity: {int(result)}")
