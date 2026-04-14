# src/preprocessing.py

import pandas as pd
import numpy as np

def create_features(df, start_date="2024-01-01"):
    """
    Crée les features (variables décalées et temporelles) nécessaires au modèle.
    """
    df = df[start_date:].copy()
    
    # Lags de la charge nette
    df["net_load_24"] = df["net_load"].shift(24)
    df["net_load_25"] = df["net_load"].shift(25)
    df["net_load_26"] = df["net_load"].shift(26)
    
    # Énergies renouvelables et leurs lags
    df["DA_renewable"] = df["DA_solar"] + df["DA_wind"]
    df["DA_renewable_1"] = df["DA_renewable"].shift(1)
    df["DA_renewable_2"] = df["DA_renewable"].shift(2)
    df["DA_renewable_3"] = df["DA_renewable"].shift(3)
    
    # Lags de la prévision de charge
    df["DA_load_1"] = df["DA_load"].shift(1)
    df["DA_load_2"] = df["DA_load"].shift(2)
    df["DA_load_3"] = df["DA_load"].shift(3)
    
    # Variables cycliques
    hours = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    
    # Liste des features
    input_features = [
        "net_load_24", "net_load_25", "net_load_26", 
        "DA_renewable", "DA_renewable_1", "DA_renewable_2", "DA_renewable_3", 
        "DA_load", "DA_load_1", "DA_load_2", "DA_load_3", 
        "hour_sin", "hour_cos"
    ]
    
    # Nettoyage des valeurs manquantes (NA générés par les shift)
    df_clean = df.dropna(subset=["net_load"] + input_features)
    
    return df_clean, input_features

def split_train_test(df_clean, split_date='2024-06-01'):
    """
    Sépare le jeu de données en set d'entraînement et de test.
    """
    df_train = df_clean[:split_date]
    df_test = df_clean[split_date:]
    return df_train, df_test