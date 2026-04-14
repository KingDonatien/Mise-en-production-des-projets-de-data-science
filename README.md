# Mise-en-production-des-projets-de-data-science
create/activate virtual environment .venv 
pip install -r requirements
run run_training.py to train and save a quantile_model.pickle in the trained_model folder
python -m uvicorn src.api:app --reload  
go on http://127.0.0.1:8000/docs