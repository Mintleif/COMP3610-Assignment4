#Imports required libraries
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import mlflow
import pandas as pd
import uuid
import time
import os
from typing import List
import joblib

#Initializes global variables for model persistence and tracking uptime
ml_model = None
start_time = None

#Manages the application lifecycle to load the model exactly once at startup
#broken path meta.yml hence I changed the model loading to download the model artifacts directly and load from the local path instead of using the registry URI which was causing issues in the Docker environment. This ensures that the model is loaded correctly when the application starts, even within a containerized setup.
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, start_time
    
    # Retrieve model path from environment variable, defaulting to the new local directory
    model_path = os.getenv("MODEL_PATH", "models/taxi_tip_regressor.pkl")

    try:
        # Load the file directly from the resolved path
        ml_model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}!")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise e
    
    start_time = time.time()
    yield


#Initializes the FastAPI application instance with a lifespan handler
app = FastAPI(title="Taxi Tip Predictor", lifespan=lifespan)

#Defines the schema for input data with strict field validation and constraints
class TripFeatures(BaseModel):
    pickup_hour: int = Field(..., ge=0, le=23, description="Hour of day")
    pickup_day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: bool
    trip_duration_minutes: float = Field(..., gt=0)
    trip_speed_mph: float = Field(..., ge=0)
    log_trip_distance: float = Field(..., ge=0)
    fare_per_mile: float = Field(..., ge=0)
    fare_per_minute: float = Field(..., ge=0)
    PU_Borough: str
    DO_Borough: str
    passenger_count: int = Field(default=1, ge=1, le=9)
    trip_distance: float = Field(..., gt=0, description="Must be positive")
    fare_amount: float = Field(..., ge=0, le=500)
    total_amount: float = Field(..., ge=0)

#Defines the structure for a single prediction response
class PredictionResponse(BaseModel):
    prediction: float
    prediction_id: str
    model_version: str

#Defines the schema for batch requests allowing up to 100 records
class BatchInput(BaseModel):
    records: List[TripFeatures] = Field(..., max_length=100)

#Defines the structure for a batch prediction response
class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    processing_time_ms: float

#Intercepts all unhandled exceptions to return a clean JSON error response
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again."
        }
    )

#Processes a single trip record to return a tip prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: TripFeatures):
    #Converts the incoming Pydantic model into a Pandas DataFrame
    input_df = pd.DataFrame([input_data.model_dump()])
    
    #Generates the prediction and extracts the result from the array
    pred = ml_model.predict(input_df)[0]
    
    return PredictionResponse(
        prediction=round(float(pred), 2),
        prediction_id=str(uuid.uuid4()),
        model_version="1"
    )

#Processes multiple trip records and calculates execution time
@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchInput):
    start = time.time()
    predictions = []
    
    #Loops through each record in the batch to generate individual predictions
    for record in batch.records:
        input_df = pd.DataFrame([record.model_dump()])
        pred = ml_model.predict(input_df)[0]
        predictions.append(PredictionResponse(
            prediction=round(float(pred), 2),
            prediction_id=str(uuid.uuid4()),
            model_version="1"
        ))
    
    #Calculates total processing time in milliseconds
    elapsed = (time.time() - start) * 1000
    return BatchResponse(
        predictions=predictions,
        count=len(predictions),
        processing_time_ms=round(elapsed, 2)
    )

#added as curl was not working in the Docker health check, this endpoint provides a simple way to verify that the API is running and responsive. It can be used for basic monitoring and debugging purposes.
# Root route to satisfy Docker's default health check and provide a simple status message
@app.get("/")
def root():
    return {"status": "alive", "message": "Taxi Tip Prediction API is operational"}

#Returns the current operational status and model availability
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model loaded": ml_model is not None,
        "model_version": "1",
        "uptime_seconds": round(time.time() - start_time, 1)
    }

#Provides metadata regarding the features and performance metrics of the model
@app.get("/model/info")
def model_info():
    return {
        "model name": "taxi-tip-regressor",
        "version": "1",
        "features": ["pickup_hour", "trip_distance", "fare_amount", "passenger_count"],
        "metrics": {"R2": 0.648, "MAE": 1.191}
    }