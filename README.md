# Energy Reserve Management

Probabilistic forecasting model for electricity reserve dimensioning in energy markets.

## Project Overview

This project implements a quantile regression approach to forecast and dimension electricity reserves in day-ahead energy markets. The model generates prediction intervals to quantify forecast uncertainty and determines optimal reserve requirements based on risk levels.

## Architecture

### Modular Structure

```
energy-reserve-mgmt/
├── config.py                     # Path and configuration
├── __init__.py                  # Package initialization
├── _quarto.yml                  # Quarto configuration
├── index.qmd                    # Website/report content
├── slides.qmd                   # Presentation content
├── notebooks/                   # Jupyter notebooks
│   ├── 1_data_import.ipynb
│   ├── 2_input_features.ipynb
│   ├── 3_descriptive_stat.ipynb
│   └── 4_fit_results.ipynb
├── data/                        # Data directory
├── figures/                     # Output plots
├── docker/                      # Production container
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── main.py                  # Entry point
│   ├── preprocessing.py        # Data preprocessing
│   ├── model.py                # Model logic
│   ├── plotting.py             # Visualization
│   ├── schemas.py              # Data schemas
│   └── static/                 # Static web assets
└── .github/workflows/          # CI/CD pipelines
    ├── jekyll-gh-pages.yml
    └── jekyll-docker.yml
```

### Key Modules

| Module | Description |
|--------|-------------|
| `preprocessing.py` | Data cleaning and feature engineering |
| `model.py` | Quantile regression model training and inference |
| `plotting.py` | Visualization and result plotting |
| `main.py` | Orchestration and API entry point |
| `schemas.py` | Pydantic data validation |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Development

Run the notebooks in order:
1. `notebooks/1_data_import.ipynb` - Load data from S3
2. `notebooks/2_input_features.ipynb` - Feature engineering
3. `notebooks/3_descriptive_stat.ipynb` - Exploratory analysis
4. `notebooks/4_fit_results.ipynb` - Model training & evaluation

### Production (Docker)

```bash
cd docker
docker-compose up --build
```

## Outputs

### Website

The project includes a Quarto-powered website (`index.qmd`) providing:
- Full methodology documentation
- Interactive results visualization
- Model performance metrics

Preview: `quarto render index.qmd --to html`

### Slides

Presentation slides (`slides.qmd`) for stakeholders:
- Business context and problem statement
- Methodology overview
- Key results and recommendations

Preview: `quarto render slides.qmd --to revealjs`

## Methodology

### Features
- Net load lags: `net_load_24`, `net_load_25`, `net_load_26`
- Renewable forecasts: `DA_solar`, `DA_wind` + lags
- Load forecast: `DA_load` + lags
- Cyclical time encoding (sin/cos)

### Model
- **Algorithm**: Quantile Regression
- **Selection**: Backward elimination (p < 5%)
- **Quantiles**: 13 evenly spaced (0.05 - 0.95)

### Validation
- **Train**: 2024 data
- **Test**: 2025 data
- **Calibration**: PIT histogram + chi-square test

## Data

| Dataset | Description |
|---------|-------------|
| `df_raw.csv` | Raw electricity data (load, renewables) |
| `df.csv` | Processed features |
| `df_procured_reserves.csv` | Reserve prices (EUR/MW) |

## Deployment

### CI/CD

GitHub Actions workflows automated:
- **GH Pages**: Website deployment on push to main
- **Docker**: Container build and push on tag

### Docker Service

The `docker/` directory contains:
- Flask API (`main.py`)
- HealthCheck endpoint
- Serialized model and preprocessing

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib
- statsmodels, scipy
- s3fs (S3 access)
- quarto (documentation)

## License

MIT License - See LICENSE file.