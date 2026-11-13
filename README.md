# Spotify Tracks Popularity Prediction

## 1. Project Overview

This project explores the use of machine learning regression models to predict the popularity of Spotify tracks based on available track and audio-related features.

The target variable is **`popularity`**, a numerical score representing a track's popularity. The objective is to investigate whether track characteristics and metadata can provide useful information for estimating this score.

The project is developed as part of the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).

### Project Objectives

- Explore the Spotify tracks dataset and identify relevant features.
- Perform data cleaning and feature engineering.
- Train and compare different regression models.
- Evaluate model performance using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
- Improve model performance through cross-validation and hyperparameter tuning.
- Package the trained model as a prediction service using FastAPI.
- Containerize the application using Docker.

## 2. Dataset

The project uses the Spotify tracks dataset, stored locally at:

```text
data/spotify_tracks.csv
```

**Dataset source:** [Spotify Tracks: Audio Features (50K Songs)](https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k?resource=download)

The dataset contains approximately 50,000 records. Initial exploration identified approximately 2,400 unique track names, indicating that different artists, albums, or versions may share the same track name.

**Note:** The dataset is not included in this GitHub repository. Users should download it from the original Kaggle source and place it in the `data/` directory.

### Target Variable

| Variable | Description |
|---|---|
| `popularity` | Numerical popularity score to be predicted |

### Feature Engineering

Several binary features are extracted from `track_name` to identify different track versions:

| Feature | Description |
|---|---|
| `keyword = Remix` | Track name contains "Remix" |
| `keyword = Live` | Track name contains "Live" |
| `keyword = Acoustic Version` | Track name contains "Acoustic Version" |
| `keyword = Radio Edit` | Track name contains "Radio Edit" |
| `keyword = feat.` | Track name contains "feat." |

The original `track_name`, `track_id`, and `album_name` columns are excluded from model training because they are identifiers or descriptive text rather than generalizable numerical features. `release_year` is also excluded from the current model.

Categorical variables are converted into numerical representations using Scikit-learn's `DictVectorizer`.

## 3. Exploratory Data Analysis

The initial EDA focuses on:

- Checking missing values, unique values, and data types.
- Investigating repeated track names.
- Identifying keywords that may indicate different track versions.
- Examining feature distributions and potential long-tailed distributions.
- Evaluating correlations between numerical features and popularity.

### Initial Findings

- No missing values were identified during the initial missing-value check.
- Some track names occur repeatedly across different artists and albums. Therefore, repeated names do not necessarily represent duplicate recordings.
- Track-version keywords were selected for additional feature engineering.
- Initial numerical correlation analysis indicates generally weak linear relationships between individual numerical features and popularity.

These findings motivate the comparison of linear and tree-based regression models.

## 4. Model Development and Evaluation

### Dataset Split

The dataset is divided into three subsets:

| Dataset | Proportion |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Test | 20% |

The split uses `random_state=1` to ensure reproducibility.

### Models Evaluated

Three regression models were initially evaluated:

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

### Evaluation Metrics

**Root Mean Squared Error (RMSE)** is used as the primary evaluation metric.

RMSE measures the average magnitude of prediction errors, with larger errors receiving a greater penalty.

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

**Mean Absolute Error (MAE)** is also used to measure the average absolute difference between actual and predicted popularity scores.

A lower RMSE or MAE indicates better prediction performance.

### Initial Model Comparison

The following results are from the initial training and validation experiments using default model configurations.

| Model | Training RMSE | Validation RMSE |
|---|---:|---:|
| Linear Regression | 17.85 | 17.83 |
| Random Forest | 6.82 | 18.10 |
| XGBoost | 15.53 | 18.27 |

**Initial Observations**

- Linear Regression achieved the lowest validation RMSE among the three initial models.
- Random Forest showed a substantial difference between training and validation RMSE, suggesting overfitting.
- XGBoost did not outperform Linear Regression with its default configuration.

### Hyperparameter Tuning

Randomized Search with 3-fold cross-validation was subsequently applied to Random Forest and XGBoost to explore alternative hyperparameter configurations.

| Model | Best Cross-Validation RMSE |
|---|---:|
| Random Forest | **17.89** |
| XGBoost | 17.93 |

The tuned Random Forest achieved the lowest cross-validation RMSE and was selected as the final model.

**Selected Random Forest Hyperparameters**

| Parameter | Value |
|---|---|
| `n_estimators` | 500 |
| `max_depth` | 30 |
| `min_samples_leaf` | 1 |
| `max_features` | `sqrt` |
| `random_state` | 1 |

### Final Model Evaluation

After model selection, the selected Random Forest model was evaluated on the held-out test dataset.

| Metric | Test Result |
|---|---:|
| RMSE | **18.13** |
| MAE | **13.51** |

The final deployment model is subsequently retrained using the full dataset and saved together with its fitted `DictVectorizer` as `model.bin`.

The reported test metrics correspond to the held-out evaluation performed before this final full-dataset training step.

## 5. Model Deployment

The selected model is packaged as a prediction service using Python and FastAPI.

### Project Components

| File | Description |
|---|---|
| `train.py` | Loads the dataset, performs feature engineering, trains the final Random Forest model, and saves the model artifacts. |
| `predict.py` | Loads the trained model and vectorizer, preprocesses incoming track data, and generates predictions. |
| `app.py` | Provides REST API endpoints using FastAPI. |
| `model.bin` | Locally generated serialized Random Forest model and fitted DictVectorizer. |

### FastAPI Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Checks whether the prediction model is available. |
| `/predict` | POST | Accepts track information and returns the predicted popularity score. |
| `/docs` | GET | Provides interactive API documentation through Swagger UI. |

The prediction API accepts a JSON object containing track information and returns a numerical popularity prediction.

Example response:

```json
{
    "predicted_popularity": 42.73
}
```

The numerical value above is illustrative.

### Docker

Docker is used to package the FastAPI application, trained model, and required dependencies into a container.

The trained model is generated locally using `train.py` before building the Docker image. This avoids storing large model artifacts in the GitHub repository.

## 6. Repository Structure

The repository is organized as follows:

```text
spotify-tracks-popularity/
│
├── data/
│   └── spotify_tracks.csv       # Download separately
│
├── notebook.ipynb
│
├── train.py
├── predict.py
├── app.py
│
├── pyproject.toml
├── uv.lock
├── requirements.txt
│
├── Dockerfile
├── .dockerignore
├── .gitignore
│
└── README.md
```

The following files are intentionally excluded from Git version control:

- `data/spotify_tracks.csv`: The original dataset is available from Kaggle.
- `model.bin`: The trained model is generated locally by running `train.py`.

This keeps the repository lightweight while allowing users to reproduce the model using the provided training script.

Dependency management is handled using [uv](https://docs.astral.sh/uv/):

- `pyproject.toml` defines the project dependencies.
- `uv.lock` records the resolved package versions.
- `requirements.txt` is exported from uv-managed dependencies for Docker installation.

## 7. How to Run

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Docker (for containerized deployment)

### Step 1: Clone the Repository

```bash
git clone https://github.com/cvtflam/03_ML-Zoomcamp_Midterm-Project
cd spotify-tracks-popularity
```

### Step 2: Download the Dataset

Download the dataset from:

[Spotify Tracks: Audio Features (50K Songs)](https://www.kaggle.com/datasets/lorenzoscaturchio/spotify-tracks-audio-features-50k?resource=download)

Create the `data/` directory if it does not exist and place the CSV file inside it:

```text
data/spotify_tracks.csv
```

### Step 3: Install Dependencies

Install the project dependencies using uv:

```bash
uv sync
```

### Step 4: Train the Model

Run the training script:

```bash
uv run python train.py
```

This trains the Random Forest model using the full dataset and generates the serialized model artifact:

```text
model.bin
```

**Note:** `model.bin` is generated locally and is not tracked by Git.

### Step 5: Run the Prediction API Locally

Start the FastAPI application:

```bash
uv run uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Use the `/predict` endpoint to submit track information and obtain a popularity prediction.

### Step 6: Run with Docker

Before building the Docker image, ensure that `model.bin` has been generated by running `train.py`.

Export the dependencies managed by uv:

```bash
uv export --format requirements.txt --output-file requirements.txt
```

Build the Docker image:

```bash
docker build -t spotify-popularity-api .
```

Run the container:

```bash
docker run --rm -p 8000:8000 spotify-popularity-api
```

The API can then be accessed through:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## 8. Project Status and Future Work

### Completed

- Dataset exploration and initial EDA.
- Data cleaning and feature engineering.
- Training and comparison of regression models.
- Randomized hyperparameter search with cross-validation.
- Final model selection and held-out test evaluation.
- Model training and serialization using `train.py`.
- Prediction pipeline using `predict.py`.
- FastAPI prediction service.
- Docker containerization.

### Potential Improvements

- Further feature engineering using artist, album, and release-related information.
- Exploring additional regression algorithms and feature selection techniques.
- Improving prediction performance through more systematic hyperparameter optimization.
- Deploying the API to a cloud platform and providing a publicly accessible prediction endpoint.

---

**Deployment URL:** To be added if cloud deployment is completed.

**API documentation:** Available locally at `http://127.0.0.1:8000/docs` when the FastAPI application is running.