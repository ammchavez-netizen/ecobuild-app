# EcoBuild — Streamlit Prototype

This app compares two early-stage building designs. For each design it predicts **heating load**, **cooling load**
and **total load** (kWh/m²) using the tuned Gradient Boosting models from the EcoBuild notebook. It then uses
**SHAP** to explain which design features cause the difference between the two designs.

## Files

| Path | What it is |
|---|---|
| `app.py` | The Streamlit app (4 tabs: Compare designs · What drives energy use · Model performance · About the data) |
| `ecobuild_core.py` | Feature engineering, model loading, prediction, SHAP (the same logic as the notebook) |
| `models/` | Model bundle exported by notebook Section 15.2 |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Theme |

## Run it on your computer

```bash
cd ecobuild_app
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501.

## Use the models trained in your own Colab run

1. Run the notebook through Section 15.2, then download `ecobuild_models.zip`.
2. Unzip it into this folder's `models/` folder, replacing the files already there.

If the scikit-learn version on your computer differs from the one used in Colab, the saved models may not load.
In that case the app automatically **retrains** them from `models/ENB2012_data.csv`, using the same split and
tuned settings (this takes a few seconds), and shows a note in the sidebar. The predictions match the notebook.

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a GitHub repository. It must include the `models/` folder.
2. Go to https://share.streamlit.io, choose **Create app**, pick the repository, and set the main file to `app.py`.
3. Click **Deploy**. Anyone with the link can then use the app.

## Notes

- Choose the building form from the 12 forms in the dataset. The models were trained on only these forms, so
  arbitrary combinations of areas and height would be extrapolation.
- If you select 0% glazing, glazing distribution is automatically set to "None", as it is in the dataset.
- The dataset is simulated for a single climate. Use the predictions to compare design options, not to forecast energy bills.
