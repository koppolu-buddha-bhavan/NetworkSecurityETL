import os,sys,json
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
import certifi
ca = certifi.where()

import pandas as pd
import numpy as np
import pymongo
from src.logger import logging
from src.exception import CustomException

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)
    
    def csv_to_json_converter(self,file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True,inplace=True)
            records = list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise CustomException(e,sys)
    
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database = database
            self.records = records
            self.collection = collection
            self.mongodb_client = pymongo.MongoClient(MONGO_DB_URL)
            
            self.database = self.mongodb_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(records)
            return len(self.records)

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':
    FILE_PATH = "Network_Data/phisingData.csv"
    DATABASE = "SURYA"
    COLLECTION = "NetworkData"
    obj = NetworkDataExtract()
    records = obj.csv_to_json_converter(FILE_PATH)
    print(records)
    print(obj.insert_data_mongodb(records,DATABASE,COLLECTION))