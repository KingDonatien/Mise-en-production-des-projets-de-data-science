python -m venv .venv
source .venv/bin/activate      #  Windows: .venv\Scripts\activate
pip install -r requirements.txt


ouvrir docker puis dans le terminal: docker compose up --build
URL: http://localhost:8000/docs
train the model by uploading a csv
then go to http://localhost:8000/static/plots/b83deb5bed404bc39fa5ec05c22c87cd/forecast_test.png to visualize the forecast

other graphs: http://localhost:8000/static/plots/b83deb5bed404bc39fa5ec05c22c87cd/forecast_train.png
http://localhost:8000/static/plots/b83deb5bed404bc39fa5ec05c22c87cd/pit_train.png
http://localhost:8000/static/plots/b83deb5bed404bc39fa5ec05c22c87cd/pit_test.png
http://localhost:8000/static/plots/b83deb5bed404bc39fa5ec05c22c87cd/cost_reliability.png
