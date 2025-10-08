from datetime import datetime
import os
from src.constants import train_pipeline 

class TrainingPipelineConfig:
    def __init__(self,timestamp = datetime.now()):
        timestamp = timestamp.strftime("%d_%m_%Y_%H_%M_%S")
        self.pipeline_name = train_pipeline.PIPELINE_NAME
        self.artifact_name = train_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name,timestamp)
        self.timestamp:str = timestamp

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir:str=os.path.join(
            training_pipeline_config.artifact_dir,train_pipeline.DATA_INGESTION_DIR
        )
        self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, train_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, train_pipeline.FILE_NAME
            )
        self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, train_pipeline.DATA_INGESTION_INGESTED_DIR, train_pipeline.TRAIN_DATA
            )
        self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, train_pipeline.DATA_INGESTION_INGESTED_DIR, train_pipeline.TEST_DATA
            )
        self.train_test_split_ratio: float = train_pipeline.DATA_INGESTION_TEST_SIZE
        self.collection_name: str = train_pipeline.DATA_INGESTION_COLLECTION
        self.database_name: str = train_pipeline.DATA_INGESTION_DATABASE

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join( training_pipeline_config.artifact_dir, train_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, train_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, train_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, train_pipeline.TRAIN_DATA)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, train_pipeline.TEST_DATA)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, train_pipeline.TRAIN_DATA)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, train_pipeline.TEST_DATA)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            train_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            train_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )

class DataTransformationConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir,train_pipeline.DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,train_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            train_pipeline.TRAIN_DATA.replace("csv", "npy"),)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  train_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            train_pipeline.TEST_DATA.replace("csv", "npy"), )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, train_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            train_pipeline.PREPROCESSING_OBJECT_FILE_NAME,)

class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, train_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, train_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            train_pipeline.MODEL_FILE_NAME
        )
        self.model_report_file_path = os.path.join(
            self.model_trainer_dir,train_pipeline.MODELS_REPORT_NAME
        )
        self.expected_accuracy: float = train_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = train_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD

