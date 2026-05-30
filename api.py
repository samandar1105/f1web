from fastapi import FastAPI
import joblib
import pandas as pd
import joblib

features = joblib.load("feature_names.pkl")

print(features)
print(len(features))
import joblib

model = joblib.load("final_model.pkl")

print(model.feature_names_in_)
print(len(model.feature_names_in_))

app = FastAPI()

model = joblib.load("final_model.pkl")

@app.get("/")
def home():
    return {"message": "F1 Pit Strategy API Running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction)
    }