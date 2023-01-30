from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from datetime import datetime
import time
import json

from classifier import NewsCategoryClassifier


class PredictRequest(BaseModel):
    source: str
    url: str
    title: str
    description: str


class PredictResponse(BaseModel):
    scores: dict
    label: str


MODEL_PATH = "../data/news_classifier.joblib"
LOGS_OUTPUT_PATH = "../data/logs.out"
NEWS_CLASSIFIER = None
LOG_FILE = None

app = FastAPI()


@app.on_event("startup")
def startup_event():
    """
    1. Initialize an instance of `NewsCategoryClassifier`.
    2. Load the serialized trained model parameters (pointed to by `MODEL_PATH`) into the NewsCategoryClassifier you initialized.
    3. Open an output file to write logs, at the destimation specififed by `LOGS_OUTPUT_PATH`
        
    Access to the model instance and log file will be needed in /predict endpoint, make sure you
    store them as global variables
    """
    global NEWS_CLASSIFIER, LOG_FILE
    news_classifier = NewsCategoryClassifier(verbose=True)
    news_classifier.load(MODEL_PATH)
    NEWS_CLASSIFIER = news_classifier
    LOG_FILE = open(LOGS_OUTPUT_PATH, "a")
    logger.info("Setup completed")


@app.on_event("shutdown")
def shutdown_event():
    """
    1. Make sure to flush the log file and close any file pointers to avoid corruption
    2. Any other cleanups
    """
    LOG_FILE.flush()
    LOG_FILE.close()
    logger.info("Shutting down application")


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    1. run model inference and get model predictions for model inputs specified in `request`
    2. Log the following data to the log file (the data should be logged to the file that was opened in `startup_event`)
    {
        'timestamp': <YYYY:MM:DD HH:MM:SS> format, when the request was received,
        'request': dictionary representation of the input request,
        'prediction': dictionary representation of the response,
        'latency': time it took to serve the request, in millisec
    }
    3. Construct an instance of `PredictResponse` and return
    """
    start_time = time.time()
    req_dict = request.dict()

    # Get prediction
    # NOTE: We shouldn't actually call predict twice on the classifier
    # - I'm just doing it to test both functions in the class.
    pred_label = NEWS_CLASSIFIER.predict_label(req_dict)
    pred_scores = NEWS_CLASSIFIER.predict_proba(req_dict)
    response = PredictResponse(scores=pred_scores, label=pred_label)

    # Log outputs
    duration = (time.time() - start_time) * 1000 # in miliseconds
    LOG_FILE.write(json.dumps({
        'timestamp': datetime.now().strftime("%Y:%m:%d %H:%M:%S"),
        'request': req_dict,
        'prediction': response.dict(),
        'latency': duration,
    }) + "\n")
    LOG_FILE.flush()
    return response


@app.get("/")
def read_root():
    return {"Hello": "World"}
