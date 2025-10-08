import sys,os
from src.logger import logging
from src.exception import CustomException

from src.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from src.constants.train_pipeline import TARGET_COLUMN
from src.constants.train_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from src.utils.main_utils.utils import save_numpy_array,save_pickle_file

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact = data_validation_artifact
            self.data_transformation_config:DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def read_file(filepath:str)->pd.DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_data_transformer_object(cls)->Pipeline:
        logging.info("Entering get_data_transformer_object")
        try:
            imputer:KNNImputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info("KNN imputer created")

            processor:Pipeline = Pipeline([("imputer",imputer)])
            return processor
        except Exception as e:
            raise CustomException(e,sys)

    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Data transformation initiated.....")
        try:
            train_df = DataTransformation.read_file(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_file(self.data_validation_artifact.vald_test_file_path)
            y_train = train_df[TARGET_COLUMN]
            y_train = y_train.replace(-1,0)
            X_train = train_df.drop(TARGET_COLUMN,axis=1)
            y_test = test_df[TARGET_COLUMN]
            y_test = y_test.replace(-1,0)
            X_test = test_df.drop(TARGET_COLUMN,axis=1)

            processor = self.get_data_transformer_object()
            transformed_X_train = processor.fit_transform(X_train)
            transformed_X_test = processor.transform(X_test)

            train_array = np.c_[transformed_X_train,np.array(y_train)]
            test_array = np.c_[transformed_X_test,np.array(y_test)]

            save_numpy_array(self.data_transformation_config.transformed_train_file_path,data=train_array)
            save_numpy_array(self.data_transformation_config.transformed_test_file_path,data=test_array)
            save_pickle_file(self.data_transformation_config.transformed_object_file_path,pkl_file=processor)

            save_pickle_file("final_model/preprocessor.pkl",processor)

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e,sys)
        




