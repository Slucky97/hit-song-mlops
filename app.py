# app.py — API qui sert le modèle de prédiction de hits
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. On crée l'application
app = FastAPI(title="Hit Song Predictor")

# 2. On charge le modèle UNE fois, au démarrage
model = joblib.load("model.joblib")

# 3. On définit la forme des données attendues en entrée (les features Spotify)
class SongFeatures(BaseModel):
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    time_signature: int

# 4. Un endpoint de santé (pour vérifier que l'API tourne)
@app.get("/")
def home():
    return {"message": "Hit Song Predictor API is running"}

# 5. L'endpoint de prédiction
@app.post("/predict")
def predict(features: SongFeatures):
    # On transforme les features reçues en DataFrame (format attendu par le pipeline)
    input_df = pd.DataFrame([features.dict()])
    # On prédit
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0].max()
    # On renvoie un résultat lisible
    result = "hit" if prediction == 1 else "non-hit"
    return {
        "prediction": result,
        "raw_prediction": int(prediction),
        "confidence": round(float(proba), 3)
    }