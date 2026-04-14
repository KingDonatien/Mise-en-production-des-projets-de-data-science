
import pandas as pd

def predict_net_load(model_result, df_test):
    """
    Effectue des prédictions sur le set de test en utilisant le modèle entraîné.
    """
    predictions = model_result.predict(df_test)
    return predictions

def evaluate_coverage(df_test, predictions):
    """
    Calcule et retourne la couverture observée (proportion de fois où la 
    charge nette réelle est inférieure ou égale à la prédiction).
    """
    coverage = (df_test['net_load'] <= predictions).mean()
    print(f"Couverture observée sur le set de test: {coverage * 100:.2f}%")
    return coverage