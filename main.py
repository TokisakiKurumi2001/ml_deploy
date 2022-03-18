from fastapi import FastAPI
from pydantic import BaseModel

import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load model and stuff
with open('models_materials/rf_model.pkl', 'rb') as file:
    clf = pickle.load(file)

with open('models_materials/scaler_feature.pkl', 'rb') as file:
    scaler = pickle.load(file)

weather_cond = ['Clear', 'Cloud', 'Sunny', 'Rainy']
# End of model and stuff


class WeatherData(BaseModel):
    temperature: float
    wind_speed: float
    pressure: int
    humidity: int
    vis_km: int
    cloud: int


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/api/v1/")
def predict(weather_data: WeatherData):
    X = np.array([weather_data.temperature, weather_data.wind_speed, weather_data.pressure,
                 weather_data.humidity, weather_data.vis_km, weather_data.cloud], dtype=np.float32).reshape(1, -1)
    res = clf.predict(scaler.transform(X))
    return {'Condition': weather_cond[res.item()]}
