
# Student Retention Dashboard

**Objective:** Predict student (or customer) retention risk and explain drivers to support interventions.

**Tech Stack:** Python, scikit-learn, SHAP, Streamlit

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train a toy model on synthetic data
python -m src.train

# Launch dashboard
streamlit run app/streamlit_app.py
```
