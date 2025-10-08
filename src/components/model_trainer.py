import sys,os
from src.logger import logging
from src.exception import CustomException
import json
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact,ClassificationMetricArtifact
from src.utils.main_utils.utils import read_numpy_array,save_json,read_pickle_file,save_pickle_file,evaluate_models
from src.utils.ML_utils.metrics.classification_metric import get_classification_score
from src.utils.ML_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from xgboost import XGBClassifier
import mlflow

class ModelTrainer():
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise CustomException(e,sys)
    
    def track_mlflow(self,best_model,classification_metric:ClassificationMetricArtifact):
        try:
            with mlflow.start_run():
                mlflow.log_metric("f1_score",classification_metric.f1_score)
                mlflow.log_metric("precision_score",classification_metric.precision_score)
                mlflow.log_metric("recall_score",classification_metric.recall_score)

                mlflow.sklearn.log_model(best_model,"model")

        except Exception as e:
            raise CustomException(e,sys)
    
    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            models = {
                "Logistic Regression": LogisticRegression(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "xgboost": XGBClassifier(
                     objective="binary:logistic",
                     eval_metric="logloss",
                     random_state=42
                   )
            }
            params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,128,256]
            },
            "xgboost":{}
            }
            ans:List = evaluate_models(X_train,y_train,X_test,y_test,params,models)

            model_report = ans[0]

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            y_train_pred = best_model.predict(X_train)

            train_metric = get_classification_score(y_train,y_train_pred)

            y_test_pred = best_model.predict(X_test)

            test_metric = get_classification_score(y_test,y_test_pred)

            self.track_mlflow(best_model,test_metric)

            preprocessor = read_pickle_file(self.data_transformation_artifact.transformed_object_file_path)
            network_model = NetworkModel(model=best_model,preprocessor=preprocessor)

            file_folder = os.path.dirname(self.data_transformation_artifact.transformed_object_file_path)
            os.makedirs(file_folder,exist_ok=True)
            
            save_pickle_file(self.model_trainer_config.trained_model_file_path,network_model)

            save_pickle_file("final_model/model.pkl",best_model)
            
            save_json(self.model_trainer_config.model_report_file_path,ans[1])

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

            return model_trainer_artifact
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def initialise_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_array = read_numpy_array(train_file_path)
            test_array = read_numpy_array(test_file_path)

            X_train = train_array[:,:-1]
            y_train = train_array[:,-1]
            X_test = test_array[:,:-1]
            y_test = test_array[:,-1]
            
            model_trainer_artifact = self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
                      
              
    

