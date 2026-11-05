# EcoBuild — Building Energy Efficiency Predictor

EcoBuild compares early-stage building designs. For each design it predicts **heating load**, **cooling load**
and **total load** (kWh/m²) with Gradient Boosting models, shows a typical error range for every prediction,
explains the differences with **SHAP**, and suggests orientation and window changes for a chosen building form.

This repository also contains the notebooks behind the research paper
*Testing on Unseen Designs: How Validation Strategy Affects Reported Accuracy in Building Energy Load Prediction*.

## Files

| Path | What it is |
|---|---|
| `app.py` | The Streamlit app (6 tabs: Compare designs · Suggest improvements · What drives energy use · Model performance · About the data · Roadmap) |
| `ecobuild_core.py` | Feature engineering, model loading, prediction, SHAP and design suggestions |
| `models/` | Model bundle used by the app (trained and tested on whole held-out designs) |
| `notebooks/EcoBuild_Paper_Figures.ipynb` | Reproduces every figure and table in the research paper |
| `notebooks/EcoBuild_Update3_Grouped_Validation.ipynb` | Whole-design validation, model tuning and the bootstrap comparison; exports the model bundle |
| `dataset/ENB2012_data.xlsx` | The UCI Energy Efficiency dataset |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Theme |

## Reproduce the research paper

1. Open `notebooks/EcoBuild_Paper_Figures.ipynb` in Google Colab or Jupyter.
2. In Colab, upload `ENB2012_data.xlsx` when asked. In Jupyter, copy `ENB2012_data.xlsx`
