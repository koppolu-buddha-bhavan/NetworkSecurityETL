import os,sys

from src.logger import logging
from src.exception import CustomException

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.utils.main_utils.utils import read_yaml,write_yaml
from src.constants.train_pipeline import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        try:
             self.data_validation_config = data_validation_config
             self.data_ingestion_artifact = data_ingestion_artifact
             self.schema = read_yaml(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
    
    def validate_number_of_columns(self,df:pd.DataFrame)->bool:
        col_num = len(self.schema["columns"])
        logging.info(f"The data should have {col_num} columns.")
        logging.info(f"The data has {len(df.columns)} columns.")
        return col_num==len(df.columns)
    
    def validate_numerical_columns(self,df:pd.DataFrame)->bool:
        numerical_col = len(self.schema["numerical_columns"])
        logging.info(f"The data should have {numerical_col} numerical columns.")
        data_numerical_col = len([col for col in df.columns if df[col].dtype == "int64"])
        logging.info(f"The data has {data_numerical_col} columns.")
        return numerical_col==data_numerical_col
    
    def detect_data_drift(self,base_data,current_data,threshold=0.05)->bool:
        try:
            status = True
            report = {}
            for col in base_data.columns:
                d1 = base_data[col]
                d2 = current_data[col]
                is_same = ks_2samp(d1,d2)
                if threshold<=is_same.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({col:{
                    "p_value":float(is_same.pvalue),
                    "drift_status":is_found
                    
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            
            dir_name = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_name,exist_ok=True)

            write_yaml(drift_report_file_path,report)

            return status

        except Exception as e :
            raise CustomException(e,sys)
        
    def initialise_data_validation(self):
        try:
            test_file_path = self.data_ingestion_artifact.test_file_path
            train_file_path = self.data_ingestion_artifact.trained_file_path

            train_data = DataValidation.read_data(test_file_path)
            test_data = DataValidation.read_data(train_file_path)

            status = self.validate_number_of_columns(train_data)
            if not status:
                error_message="Train data doesn't have all the columns!"
        
            status = self.validate_number_of_columns(test_data)
            if not status:
                error_message="Test data doesn't have all the columns!"
        
            status = self.validate_numerical_columns(train_data)
            if not status:
                error_message="Train data doesn't have all the numerical columns!"
        
            status = self.validate_numerical_columns(test_data)
            if not status:
                error_message="Test data doesn't have all the numerical columns!"
           
            status = self.detect_data_drift(train_data,test_data)
            valid_dir_name = os.path.dirname(self.data_validation_config.valid_test_file_path)
            os.makedirs(valid_dir_name,exist_ok=True)

            train_data.to_csv(self.data_validation_config.valid_train_file_path,index=False,header=True)
            test_data.to_csv(self.data_validation_config.valid_test_file_path,index=False,header=True)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                vald_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
        
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
        