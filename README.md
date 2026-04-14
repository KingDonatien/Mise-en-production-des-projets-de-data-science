# Energy Reserve Management

Probabilistic forecasting model for electricity reserve dimensioning in energy markets.

## Project Overview

This project implements a quantile regression approach to forecast and dimension electricity reserves in day-ahead energy markets. The model generates prediction intervals to quantify forecast uncertainty and determines optimal reserve requirements based on risk levels.

## Repository Structure

```
energymngmt_prod/
├── config.py              # Path configuration
├── _quarto.yml            # Quarto configuration
├── index.qmd              # Quarto report
├── slides.qmd             # Quarto slides
├── notebooks/
│   ├── 1_data_import.ipynb       # Data loading from S3
│   ├── 2_input_features.ipynb    # Feature engineering
│   ├── 3_descriptive_stat.ipynb  # Exploratory analysis
│   └── 4_fit_results.ipynb       # Model training & results
├── data/
│   ├── df_raw.csv          # Raw data
│   ├── df.csv              # Processed data with features
│   └── df_procured_reserves.csv  # Reserve cost data
└── figures/                # Output plots


## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Notebooks

Execute in this order:

1. **1_data_import.ipynb**: Load electricity data from S3 storage
2. **2_input_features.ipynb**: Create input features (lags, renewable aggregates, cyclical time features)
3. **3_descriptive_stat.ipynb**: Visualize data characteristics
4. **4_fit_results.ipynb**: Train quantile regression models, analyze results and study reserve cost


## Methodology

### Input Features

- **Net load lags**: `net_load_24`, `net_load_25`, `net_load_26` (previous day same hour and adjacent hours)
- **Renewable forecast**: `DA_solar`, `DA_wind`, and their lags
- **Load forecast**: `DA_load` and its lags
- **Cyclical time features**: Hour of day encoded as sin/cos components

### Model

- **Algorithm**: Quantile Regression (linear)
- **Feature Selection**: Backward elimination based on p-values (threshold: 5%)
- **Quantiles**: 13 evenly spaced quantiles from 0.05 to 0.95

### Validation

- **Training set**: 2024 data
- **Test set**: 2025 data
- **Calibration checks**: PIT (Probability Integral Transform) histogram and chi-square test

## Results

### Model Performance

- Prediction intervals are well-calibrated (PIT values follow uniform distribution)
- Chi-square tests confirm probabilistic calibration

### Reserve Costs

The model provides a cost-reliability trade-off:
- Lower risk tolerance ($\epsilon$) requires larger reserves but provides more reliability
- Higher risk tolerance reduces reserve costs but increases under-coverage frequency

## Data

### Input Data

Electricity data from ENTSOE including:

df_raw.csv:
- Net load (actual load - actual renewables)
- Day-ahead load forecasts
- Day-ahead renewable generation forecasts (solar, wind)
- Actual renewable generation

df_procured_reserves.csv:
- Historical procured reserve prices (EUR/MW) from European energy markets.




## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- pathlib
- statsmodels
- scipy
- sys
- s3fs (for S3 data access)

## License

This project is licensed under the MIT License. See the LICENSE file in this repository or visit MIT License for more information.

## Authors

Adrien Barrau
Younes Iggidr
Clément Morelière
Donatien Roi
