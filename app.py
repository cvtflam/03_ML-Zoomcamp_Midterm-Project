
from typing import Any

from fastapi import FastAPI, HTTPException

from predict import predict as make_prediction
from predict import load_model


# --------------------------------------------------
# 1. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Spotify Tracks Popularity Prediction API",
    description="Predict Spotify track popularity using a Random Forest model.",
    version="1.0.0"
)


# --------------------------------------------------
# 2. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    try:
        load_model()

        return {
            "status": "OK",
            "model": "Random Forest"
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run train.py first."
        )


# --------------------------------------------------
# 3. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_endpoint(track: dict[str, Any]):

    # Basic input validation
    if not isinstance(track.get("track_name"), str):
        raise HTTPException(
            status_code=422,
            detail="'track_name' must be provided as a string."
        )

    try:

        prediction = make_prediction(track)

        return {
            "predicted_popularity": round(prediction, 2)
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model file not found."
        )
