## Review : Parcours publication reproductible



Projet intéressant, la pb de prévision de charge nette est bien posée, on va jusqu'à FastAPI + docker.



## Points positifs

- bonne structure du repo (séparation entre rapport Quarto, notebooks, données et API)
- README : contexte, méthodologie, architecture et commandes principales sont là.
- Présence de LICENSE, requirements.txt, GitHub Actions, Dockerfile et docker-compose.yml.
- modularité Python avec `preprocessing.py`, `model.py`, `plotting.py` et `main.py`.
- Usage de Quarto fonctionnel



## Points d'amélioration

### 1. Workflow GitHub Actions

Le workflow utilise Jekyll alors que le rapport est en Quarto — pas sur que le site soit bien reconstruit depuis les `.qmd` automatiquement


### 2. Duplication rapport / scripts Python

Dans `index.qmd`, le feature engineering, l'entraînement et les graphiques sont recodés directement, alors que `preprocessing.py`, `model.py` et `plotting.py` existent déjà dans la partie Docker.


### 3. csv dans le repo

Les CSV sont dans `data/` 

---

### 4. Fichiers à exclure du repo

Un dossier `__pycache__` est versionné. À ajouter dans `.gitignore` :

pycache/
*.pyc
 


### 5. Incohérence schémas / endpoints

`schemas.py` définit des schémas pour `/train`, `/predict` et `/models`, mais `main.py` n'expose que `/forecast`



### 6. Persistance du modèle

C'est lié au point précédent : un volume `/models` est défini et `joblib` est là pour sauvegarder, mais dans `/forecast` le modèle est réentraîné à chaque appel

---

### 7. Tests et linter

Pas de tests ni de linter visibles. Quelques tests `pytest` sur les fonctions de preprocessing + un `ruff` ou `black` serait bien

---

## Conclusion

Bonne base : la combinaison Quarto + api + docker est bien amenée. Le plus important selon moi est la duplication de logique entre le rapport et les scripts Python (point 2) et l'incohérence entre le design de l'API et son implémentation actuelle (points 6 et 7).