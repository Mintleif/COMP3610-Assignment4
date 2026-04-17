# Assignment 4: MLOps Pipeline for NYC Taxi Tip Prediction

## Project Overview

This repository presents a fully containerized machine learning pipeline for predicting taxi tips using the NYC Taxi dataset. The system integrates:

* **MLflow** for experiment tracking and model registry
* **FastAPI** for real-time model inference
* **Docker Compose** for orchestration of all services

This architecture ensures a reproducible, scalable, and production-like deployment environment.

---

## Repository Structure

```
├── assignment4.ipynb        # End-to-end ML pipeline (training + MLflow logging)
├── app.py                  # FastAPI application for model serving
├── demo_requests.py        # Script to test API with sample requests
├── test_app.py             # Pytest suite for API validation
├── models/                 # (Optional) Local model artifacts
├── Dockerfile              # API container configuration
├── docker-compose.yml      # Multi-service orchestration (API + MLflow)
├── requirements.txt        # Python dependencies
├── .dockerignore           # Docker build optimization
└── .gitignore              # Git file exclusions
```

---

## Data Ingestion

The pipeline retrieves data programmatically to keep the repository lightweight:

* NYC Taxi Zone lookup table is downloaded from an official public source
* Processed dataset (`cleaned_taxi_2024_01.parquet`) is loaded dynamically
* Ensures reproducibility without storing large datasets locally

---

## System Architecture

The system consists of two main services:

| Service           | Description                                         |
| ----------------- | --------------------------------------------------- |
| **MLflow Server** | Tracks experiments, stores models, manages registry |
| **FastAPI API**   | Loads registered model and serves predictions       |

Both services communicate within a Docker network.

---

## Deployment Instructions

### 1. Start the System (Docker)

From the project root:

```bash
docker compose up --build
```

This will:

* Build the FastAPI container
* Start the MLflow tracking server
* Create a shared Docker network

---

### 2. Access the Services

Once running:

* **FastAPI Swagger UI**
  [http://localhost:8000/docs](http://localhost:8000/docs)

* **MLflow Dashboard**
  [http://localhost:5000](http://localhost:5000)

---

### 3. Register the Model (IMPORTANT)

After starting Docker:

1. Open `assignment4.ipynb`
2. Ensure:

```python
mlflow.set_tracking_uri("http://localhost:5000")
```

3. Run all cells to:

   * Train models
   * Log experiments
   * Register model (`taxi-tip-regressor`)

---

### 4. Restart API (after model registration)

```bash
docker compose restart api
```

---

### 5. Test the API

#### Option A: Python script

```bash
python demo_requests.py
```

#### Option B: Swagger UI

Go to:

```
http://localhost:8000/docs
```

---

## Automated Testing

Run the test suite locally:

```bash
pytest -v test_app.py
```

Tests include:

* Health endpoint validation
* Prediction correctness
* Input validation (edge cases)
* Batch predictions

---

## API Endpoints

| Endpoint         | Method | Description          |
| ---------------- | ------ | -------------------- |
| `/predict`       | POST   | Single prediction    |
| `/predict/batch` | POST   | Batch predictions    |
| `/health`        | GET    | Service health check |
| `/model/info`    | GET    | Model metadata       |

---

## Key MLOps Concepts Demonstrated

* Experiment tracking with MLflow
* Model versioning and registry
* Containerized deployment
* API-based model serving
* Automated testing and validation
* Separation of backend and artifact storage

---

## Troubleshooting

### Model not loading?

Ensure:

* MLflow is running (`http://localhost:5000`)
* Model is registered (`taxi-tip-regressor`)
* You ran the notebook **after starting Docker**

---

### API keeps restarting?

Run:

```bash
docker compose logs api
```

Common cause:

* Model not found in MLflow registry

---

## Shutdown

To stop all services:

```bash
docker compose down
```

---

## Notes

* MLflow uses:

  * `mlruns/` → metadata
  * `mlartifacts/` → model files
* Both are mounted as Docker volumes for persistence
* The system mimics a real-world production ML deployment pipeline
