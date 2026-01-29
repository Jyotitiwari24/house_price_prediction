from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from app.model import load_model

app = FastAPI()
model = load_model()


class HouseData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/")
def home():
    return {"message": "House Price Prediction API"}


@app.post("/predict")
def predict(data: HouseData):
    features = np.array([[ 
        data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
        data.Population, data.AveOccup, data.Latitude, data.Longitude
    ]])
    prediction = model.predict(features)
    return {"predicted_price": float(prediction[0])}
