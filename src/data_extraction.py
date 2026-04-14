# src/data_extraction.py

import os
import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

# Charger les variables d'environnement situées dans le .env à la racine
load_dotenv()

def download_entsoe_data(start_date='20240101', end_date='20251230', country_code='FR', output_path='data/df.csv'):
    """
    Télécharge les données depuis l'API ENTSOE en utilisant la clé API 
    stockée dans le fichier .env.
    """
    # Récupération de la clé API depuis les variables d'environnement
    api_key = os.getenv("ENTSOE_API_KEY")
    
    if not api_key:
        raise ValueError(
            "ERREUR : La variable ENTSOE_API_KEY est introuvable. "
            "Vérifiez que votre fichier .env contient bien cette clé."
        )

    client = EntsoePandasClient(api_key=api_key)
    
    start = pd.Timestamp(start_date, tz='UTC')
    end = pd.Timestamp(end_date, tz='UTC')
    
    print(f"Téléchargement des données ENTSOE pour {country_code}...")
    
    # Requêtes API
    loads = client.query_load_and_forecast(country_code, start=start, end=end)
    wind_solar_forecast = client.query_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None)
    
    actual_gen = client.query_generation(country_code, start=start, end=end, psr_type=None)
    
    # Sélection et renommage des colonnes de génération réelle
    actual_wind_solar = actual_gen[[
        ("Solar", "Actual Aggregated"), 
        ("Wind Offshore", "Actual Aggregated"), 
        ("Wind Onshore", "Actual Aggregated")
    ]]
    actual_wind_solar.columns = actual_wind_solar.columns.map('_'.join)
    
    # Jointure des données
    df = loads.join(wind_solar_forecast, how='inner')
    df = df.join(actual_wind_solar, how='inner')
    
    # Harmonisation des noms de colonnes
    df.columns = [
        "DA_load", "actual_load", "DA_solar", "DA_offshore", 
        "DA_onshore", "actual_solar", "actual_offshore", "actual_onshore"
    ]
    
    # Calcul des agrégats et de la charge nette
    df["DA_wind"] = df["DA_offshore"].fillna(0) + df["DA_onshore"].fillna(0)
    df["actual_wind"] = df["actual_offshore"].fillna(0) + df["actual_onshore"].fillna(0)                                                    
    df["net_load"] = df["actual_load"] - df["actual_solar"] - df["actual_wind"]
    
    # Sélection finale
    df = df[["net_load", "DA_load", "actual_load", "DA_solar", "DA_wind", "actual_solar", "actual_wind"]]
    df = df.asfreq('h')
    
    # Création du dossier data s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=True)
    print(f"Données sauvegardées sous {output_path}")
    return df

def load_data(filepath="data/df.csv"):
    """
    Charge les données depuis le fichier CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} n'existe pas. Lancez d'abord download_entsoe_data().")
        
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df