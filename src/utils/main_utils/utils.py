from src.logger import logging
from src.exception import CustomException

import os,sys
import yaml
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,accuracy_score,f1_score,precision_score,recall_score
import json
from typing import List


def read_yaml(file_path:str)->dict:
    try:
        with open(file_path,"rb") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise CustomException(e,sys)
    
def write_yaml(file_path,report,replace:bool=False):
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as file:
            yaml.dump(report,file)
    except Exception as e:
        raise CustomException(e,sys)

def save_numpy_array(file_path:str,data:np.array):
    try:
        logging.info("Entering folder to save the array...")
        folder = os.path.dirname(file_path)
        os.makedirs(folder,exist_ok=True)
        with open(file_path,"wb") as f:
            np.save(f,data)
        logging.info("Array saved!")
    except Exception as e:
        raise CustomException(e,sys)

def save_pickle_file(file_path:str,pkl_file:object):
    try:
        logging.info("Entering folder to save the object...")
        folder = os.path.dirname(file_path)
        os.makedirs(folder,exist_ok=True)
        with open(file_path,"wb") as f:
            pickle.dump(pkl_file,f)
        logging.info("Pickle file saved successfully!")
    except Exception as e:
        raise CustomException(e,sys)

def read_numpy_array(file_path:str)->np.array:
    try:
        logging.info("Reading numpy array....")
        with open(file_path,"rb") as f:
            logging.info("Array read successfully!")
            return np.load(f)
        
    except Exception as e:
        raise CustomException(e,sys)

def read_pickle_file(file_path:str)->object:
    try:
        logging.info("Reading pickle file....")
        with open(file_path,"rb") as f:
            logging.info("Pickle file read successfully!")
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e,sys)

def save_json(file_path:str,obj:dict):
    try:
        with open(file_path,"w") as f:
            json.dump(obj,f,indent=4)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,params,models)->List:
    try:
        report = {}
        models_report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            grid = GridSearchCV(model,param,cv=3)
            grid.fit(X_train,y_train)

            model.set_params(**grid.best_params_)

            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)

            model_r2_score = r2_score(y_true=y_test,y_pred=y_pred)
            report[list(models.keys())[i]] = model_r2_score
            sub_report = {}
            sub_report["f1_score"] = f1_score(y_test,y_pred)
            sub_report["precision_score"] = precision_score(y_test,y_pred)
            sub_report["recall_score"] = recall_score(y_test,y_pred)
            sub_report["accuracy"] = accuracy_score(y_test,y_pred)
            models_report[list(models.keys())[i]] = sub_report
            ans:List = []
            ans.append(report)
            ans.append(models_report)
        return ans
    except Exception as e:
        raise CustomException(e,sys)
