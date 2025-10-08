import os,sys
import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)
import pymongo

from src.exception import CustomException
from src.logger import logging
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.ML_utils.model.estimator import NetworkModel

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from src.utils.main_utils.utils import read_pickle_file

from src.constants.train_pipeline import DATA_INGESTION_DATABASE,DATA_INGESTION_COLLECTION

client = pymongo.MongoClient(mongo_db_url,tlsCAFile=ca)

db = client[DATA_INGESTION_DATABASE]
collection = db[DATA_INGESTION_COLLECTION]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as e:
        raise CustomException(e,sys)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory="./templates")

@app.post("/predict")
async def predict_route(request:Request,file:UploadFile=File(...)):
    try:
        df=pd.read_csv(file.file)
        #print(df)
        preprocesor=read_pickle_file("final_model/preprocessor.pkl")
        final_model=read_pickle_file("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocesor,model=final_model)
        #print(df.iloc[0])
        y_pred = network_model.predict(df)
        print(y_pred)
        df['predicted_column'] = y_pred
        print(df['predicted_column'])
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        df.to_csv('prediction_output/output.csv')
        table_html = df.to_html(classes='table table-striped')
        #print(table_html)
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    app_run(app,host="localhost",port=8000)

