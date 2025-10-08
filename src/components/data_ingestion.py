import os,sys
from src.logger import logging
from src.exception import CustomException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)
        
    def export_collection_as_dataFrame(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in list(df.columns):
                df.drop(columns=["_id"],axis=1,inplace=True)
            
            df.replace("na",np.nan,inplace=True)
            return df

        except Exception as e:
            raise CustomException(e,sys)
        
    def export_data_to_feature_store(self,dataFrame:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            raw_path = os.path.dirname(feature_store_file_path)
            os.makedirs(raw_path,exist_ok=True)
            dataFrame.to_csv(feature_store_file_path,index=False,header=True)
        except Exception as e:
            raise CustomException(e,sys)
    
    def split_data(self,dataframe:pd.DataFrame):
        try:
            train_data,test_data = train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)
            logging.info("Train test split completed.")

            dirpath = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dirpath,exist_ok=True)

            logging.info("Exporting test data and training data....")

            train_data.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info("Exported successfully!")
        
        except Exception as e:
            raise CustomException(e,sys)

    
    def initialise_data_ingestion(self):
        try:
            dataFrame = self.export_collection_as_dataFrame()
            self.export_data_to_feature_store(dataFrame)
            self.split_data(dataFrame)
            data_ingestion_artifact = DataIngestionArtifact(test_file_path=self.data_ingestion_config.testing_file_path,trained_file_path=self.data_ingestion_config.training_file_path)

            return data_ingestion_artifact        

        except Exception as e:
            raise CustomException(e,sys)
        


