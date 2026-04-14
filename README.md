# Mise-en-production-des-projets-de-data-science
create/activate .venv
pip install -r requirements
run run_training.py to save a .pickle model in folder trained_models
python -m uvicorn src.api:app --reload
go to http://127.0.0.1:8000/docs

