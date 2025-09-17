from fastapi import FastAPI, Request, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import sys
import shutil
import pandas as pd
import io


from src.exception_setup.exception import AnomalyDetectionException
from src.logging_setup import logger

from src.config_entities.config_entity import TrainPipelineConfig, DataIngestionConfig
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.inference_pipeline import ModelInference

app = FastAPI()


@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={
            "message": "👋 Welcome to the Anomaly Detection API.",
            "upload_endpoint": "/upload",
            "usage": "Send a POST request with a CSV file to /upload"
        }
    )


@app.post("/upload")
async def upload_file(request:Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        

        train_config = TrainPipelineConfig(timestamp=datetime.now())
        data_ingestion_config = DataIngestionConfig(train_config)

        upload_path = os.path.join(data_ingestion_config.data_ingestion_dir, data_ingestion_config.uploaded_dir)
        os.makedirs(upload_path, exist_ok=True)

        destination_file_path = os.path.join(upload_path, data_ingestion_config.uploaded_data_name)
        with open(destination_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        
        train_pipeline = TrainingPipeline()
        background_tasks.add_task(train_pipeline.run_pipeline)

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded successfully.",
                "file_path": os.path.abspath(destination_file_path)
            }
        )
    except Exception as e:
        raise AnomalyDetectionException(e, sys)
    


@app.post(
    "/predict/csv",
    summary="Predict anomalies from CSV",
    description="Upload a CSV file with time-series data. Returns anomaly flags per window."
)
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")
    if file.content_type not in ("text/csv", "application/vnd.ms-excel"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        
        logger.logging.info(f"Uploaded CSV shape: {df.shape}")
        logger.logging.info(f"Uploaded CSV columns: {df.columns.tolist()}")
        logger.logging.info(f"Uploaded CSV head:\n{df.head().to_string()}")

        
        inference = ModelInference()
        detector = inference.get_detector()
        results = detector.predict_from_dataframe(df)

        if not results:
            return JSONResponse(content={"message": "Not enough rows for one window", "results": []})

        if sum(r["flag"] for r in results["predictions"]) == 0:
            return JSONResponse(content={"message": "No anomalies detected", "results": results["predictions"]})

        return JSONResponse(content={"message": "OK", "results": results["predictions"]})

    except Exception as e:
        logger.logging.error(f"Prediction failed: {e}")
        raise AnomalyDetectionException(e, sys)
