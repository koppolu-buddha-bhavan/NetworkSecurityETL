from src.logger import logging
from src.exception import CustomException
import sys

from src.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score,precision_score,recall_score


def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true,y_pred)
        model_precision_score = precision_score(y_true,y_pred)
        model_recall_score = recall_score(y_true,y_pred)
        logging.info("Classification score is being calculated....")
        classification_metric_artifact = ClassificationMetricArtifact(f1_score=model_f1_score,precision_score=model_precision_score,recall_score=model_recall_score)
        logging.info("Classification score is returned!")
        return classification_metric_artifact
    except Exception as e:
        raise CustomException(e,sys)