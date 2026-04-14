# run_training.py
from src.data_extraction import load_data
from src.preprocessing import create_features, split_train_test
from src.train import train_quantile_regression

# 1. Charger le CSV local
df = load_data("data/df.csv")

# 2. Créer les features
df_clean, _ = create_features(df)

# 3. Séparer Train/Test
df_train, df_test = split_train_test(df_clean)

# 4. Entraîner et sauvegarder le modèle
train_quantile_regression(df_train, quantile=0.9, save_path="trained_models/quantile_model.pickle")