from src.constants.train_pipeline import SAVED_MODEL_DIR,MODEL_FILE_NAME
import sys,os

from src.logger import logging
from src.exception import CustomException

class NetworkModel:
    def __init__(self,model,preprocessor):
        try:
            self.model = model
            self.preprocessor = preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def predict(self,X_test):
        try:
            X_test_transformed = self.preprocessor.transform(X_test)
            logging.info("Preprocessing completed!")
            y_pred = self.model.predict(X_test_transformed)
            logging.info("Prediction completed!")
            return y_pred
        except Exception as e:
            raise CustomException(e,sys)

