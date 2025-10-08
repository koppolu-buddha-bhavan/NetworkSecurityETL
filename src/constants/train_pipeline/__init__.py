import os,sys
import pandas as pd
import numpy as np


TARGET_COLUMN: str = "Result"
PIPELINE_NAME: str = "NetworkSecurity"
FILE_NAME: str = "phisingData.csv"
ARTIFACT_DIR: str = "Artifacts"


TRAIN_DATA: str = "train.csv"
TEST_DATA: str = "test.csv"
SCHEMA_FILE_PATH: str = os.path.join("data_schema","schema.yaml")

DATA_INGESTION_DATABASE: str = "SURYA"
DATA_INGESTION_COLLECTION: str = "NetworkData"
DATA_INGESTION_DIR: str = "data_ingestion"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_FEATURE_STORE_DIR : str = "feature_store"
DATA_INGESTION_TEST_SIZE: float = 0.2

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"

MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05

SAVED_MODEL_DIR =os.path.join("saved_models")
MODEL_FILE_NAME = "model.pkl"
MODELS_REPORT_NAME = "models_report.json" 

