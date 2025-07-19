from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from fastapi import HTTPException
from pydantic import BaseModel
from typing import List

import joblib
from datetime import date
import os
import uvicorn

app = FastAPI()

MODEL_PATH = './Model/svm.joblib'
ENCODER_MODEL = "all-MiniLM-L6-v2"

# create instance of & load model & encoder outside of POST to eliminate redundancies
model = joblib.load(MODEL_PATH)
encoder = SentenceTransformer(ENCODER_MODEL)

@app.get("/status")
def status():
    """GET Status API call"""

    return JSONResponse({"status" : "ok"})

class HeadlineRequest(BaseModel):
    headlines: List[str]

@app.post("/score_headlines")
def score_headlines(request: HeadlineRequest):
    """POST API Call to score headlines"""

    if not request.headlines:
        raise HTTPException(status_code=400, detail="Please enter a headline")

    try:
        vectors = encoder.encode(request.headlines)
        predictions = model.predict(vectors)
        return JSONResponse(content={"predictions": predictions.tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unable to make predictions. Please try again. {str(e)}")


if __name__ == "__main__":
    uvicorn.run("score_headlines_api:app", host="127.0.0.1", port=8007, reload=True)