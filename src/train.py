import os
import statsmodels.formula.api as smf

def train_quantile_regression(df_train, quantile=0.9, save_path="trained_models/quantile_model.pickle"):
    """
    Entraîne un modèle de régression par quantile avec statsmodels et le sauvegarde 
    dans le dossier spécifié.
    """
  
    folder = os.path.dirname(save_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Dossier créé : {folder}")

    formula = (
        "net_load ~ net_load_24 + net_load_25 + net_load_26 + "
        "DA_renewable + DA_renewable_1 + DA_renewable_2 + DA_renewable_3 + "
        "DA_load + DA_load_1 + DA_load_2 + DA_load_3 + "
        "hour_sin + hour_cos"
    )
    
    print(f"Début de l'entraînement du modèle pour le quantile q={quantile}...")

    model = smf.quantreg(formula, data=df_train)
    result = model.fit(q=quantile)
    result.save(save_path)
    print(f"Modèle sauvegardé avec succès dans : {save_path}")
    
    return result