# src/api.py

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Variable globale pour stocker le modèle en mémoire
model = None

# Définition du schéma d'entrée attendu par l'API
class PredictInput(BaseModel):
    net_load_24: float
    net_load_25: float
    net_load_26: float
    DA_renewable: float
    DA_renewable_1: float
    DA_renewable_2: float
    DA_renewable_3: float
    DA_load: float
    DA_load_1: float
    DA_load_2: float
    DA_load_3: float
    hour_sin: float
    hour_cos: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Logique exécutée au démarrage de l'API : charger le modèle
    global model
    model_path = os.getenv("MODEL_PATH", "trained_models/quantile_model.pickle")
    try:
        model = sm.load(model_path)
        print(f"Modèle chargé avec succès depuis {model_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
    yield
    # Logique exécutée à l'extinction de l'API (nettoyage si nécessaire)
    model = None

app = FastAPI(
    title="API de Prévision de Charge Nette",
    description="API exposant le modèle de régression quantile via statsmodels",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    """Vérifie que l'API est en ligne."""
    return {"status": "ok", "message": "API opérationnelle"}

@app.post("/predict")
def predict(data: PredictInput):
    """
    Accepte les features temporelles et retourne la prédiction du quantile de charge nette.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")
    
    try:
        # Transformation du JSON entrant en DataFrame (requis par statsmodels)
        input_data = pd.DataFrame([data.model_dump()])
        
        # Prédiction
        prediction = model.predict(input_data)
        
        return {
            "prediction_net_load": float(prediction.iloc[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction: {str(e)}")