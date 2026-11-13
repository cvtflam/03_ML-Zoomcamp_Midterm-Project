from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


# 1. Configuration
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "spotify_tracks.csv"
MODEL_PATH = BASE_DIR / "model.bin"

KEYWORDS = [
    'Remix',
    'Live',
    'Acoustic Version',
    'Radio Edit',
    'feat.'
]

def preprocess_data(df):
    df = df.copy()
    # 2. Data preprocessing and feature engineering
    # Finalized keywords, create 5 new columns in the dataframe to indicate whether the track name contains any of the keywords.
    for keyword in KEYWORDS:
        df[f'keyword = {keyword}'] = df['track_name'].apply(lambda x: 1 if keyword in x else 0)

    # Since track_name, track_id, and album_name should be unique for every song, these parameters will not be used as a feature in the model.
    # Also, release_year is not related to the prediction of the popularity of a newly released track, so it will not be used as a feature in the model.
    # This dataframe will be used for the tree based model.
    df_data = df.drop(['track_name', 'track_id', 'album_name', 'release_year'], axis=1)
    for col in df_data.columns:
        if df_data[col].dtype == 'str':
            df_data[col] = df_data[col].str.lower()
            df_data[col] = df_data[col].str.strip()

    return df_data


def train():
    # Load the data
    df = pd.read_csv(DATA_PATH)

    # Preprocess the data
    df_data = preprocess_data(df)
    # 3. Prepare the data for modeling. The target variable is "popularity".
    y_train = df_data['popularity'].values
    df_train = df_data.drop('popularity', axis=1)

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train.to_dict(orient='records'))

    # 4. Train a tree based model.

    rf = RandomForestRegressor(        
        n_estimators=500,
        max_depth=30,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=1,
        n_jobs=-1

    )

    rf.fit(X_train, y_train)

    with open(MODEL_PATH, 'wb') as f_out:
        pickle.dump(
            {
                'model': rf,
                'dv': dv
            },
            f_out
        )

    print(f"Model saved to: {MODEL_PATH}")

if __name__ == '__main__':
    train()