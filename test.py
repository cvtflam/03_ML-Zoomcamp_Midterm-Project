import pandas as pd
import requests
import json

df = pd.read_csv("data/spotify_tracks.csv")

# Select one record, excluding the target column
track = json.loads(
    df.drop(columns=["popularity"])
      .iloc[0]
      .to_json()
)

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=track
)

print(response.status_code)
print(response.json())