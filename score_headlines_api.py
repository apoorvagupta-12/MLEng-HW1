"""Script to create an API to score headlines leveraging knowledge from score_headlines.py"""

import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = './Model/svm.joblib'
ENCODER_MODEL = "all-MiniLM-L6-v2"

# create instance of & load model & encoder outside of POST to eliminate redundancies
model = joblib.load(MODEL_PATH)
encoder = SentenceTransformer(ENCODER_MODEL)

@app.get("/status")
def status():
    """GET Status API call"""
    logger.info("Received GET request for /status")
    return JSONResponse({"status" : "ok"})


class HeadlineRequest(BaseModel):
    """Class to define the expected parameter type from POST request"""

    headlines: List[str]

@app.post("/score_headlines")
def score_headlines(request: HeadlineRequest):
    """POST API Call to score headlines"""
    logger.info("Received POST request for /score_headlines")


    if not request.headlines:
        logger.info("No headlines provided")
        raise HTTPException(status_code=400, detail="Please enter a headline")

    try:
        vectors = encoder.encode(request.headlines)
        predictions = model.predict(vectors)
        logger.info("Predictions made and returned")
        return JSONResponse(content={"predictions": predictions.tolist()})
    except Exception as e:
        logger.warning("Error making predictions")
        raise HTTPException(status_code=500,
                            detail=f"Unable to make predictions. Please try again. {str(e)}") from e


if __name__ == "__main__":
    uvicorn.run("score_headlines_api:app", host="127.0.0.1", port=8007, reload=True)
