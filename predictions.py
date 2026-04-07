import streamlit as st
import pickle
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
st.set_page_config(page_title="Predictions", layout="wide")
 
# Load models
with open("./models/cal_rf.pkl", "rb") as f:
    model = pickle.load(f)
 
with open("./models/bcc_rf.pkl", "rb") as f:
    bcc = pickle.load(f)
 
with open("./models/mapie_rf.pkl", "rb") as f:
    mapie = pickle.load(f)
 
# Load SHAP explainer
explainer = joblib.load("./data/shap/explainer.joblib")
 
# Load standardizer
scaler = joblib.load("./data/standardization/standard_scaler.joblib")
cols = joblib.load("./data/standardization/columns.joblib")
 
# Class labels
CLASS_LABELS = {0: "No Sleep Disorder", 1: "Sleep Apnea", 2: "Insomnia"}
 
st.title("Predictions")
st.markdown("### A Random Forest Classifier that predicts whether a patient has a Sleep Disorder or not by using:\n##### * Class Predictions\n##### * Probability Predictions\n##### * Conformal Predictions for Uncertainty Quantification\n##### * SHAP Values for Interpretability")
st.divider()
 
st.markdown("### This model uses sleep health and lifestyle data")
st.divider()
 
st.markdown("### Enter inputs")
st.divider()
 
with st.form("Patient Inputs"):
    c1, c2 = st.columns(2)
 
    with c1:
        gender      = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        age         = st.slider("Age", min_value=18, max_value=80, value=35)
        sleep_dur   = st.slider("Sleep Duration (hours)", min_value=4.0, max_value=10.0, value=7.0, step=0.1)
        quality     = st.slider("Quality of Sleep (1–10)", min_value=1, max_value=10, value=7)
 
    with c2:
        phys_act    = st.slider("Physical Activity Level (mins/day)", min_value=0, max_value=120, value=60)
        stress      = st.slider("Stress Level (1–10)", min_value=1, max_value=10, value=5)
        heart_rate  = st.slider("Heart Rate (bpm)", min_value=50, max_value=100, value=72)
        daily_steps = st.slider("Daily Steps", min_value=1000, max_value=25000, value=7000, step=500)
 
    submitted = st.form_submit_button("Run prediction")
 
if submitted:
    # Build input dataframe
    raw = pd.DataFrame([{
        "Gender": int(gender),
        "Age": int(age),
        "Sleep Duration": float(sleep_dur),
        "Quality of Sleep": int(quality),
        "Physical Activity Level": int(phys_act),
        "Stress Level": int(stress),
        "Heart Rate": int(heart_rate),
        "Daily Steps": int(daily_steps),
    }])
 
    # Engineer derived features
    raw["Heart Rate Categoricl"]               = (raw["Heart Rate"] >= 76).astype(float)
    raw["Sleep Duration Log"]                  = np.log1p(raw["Sleep Duration"])
    raw["Sleep Duration * Stress Level"]       = raw["Sleep Duration"] * raw["Stress Level"]
    raw["Physical Activity Level Categorical"] = pd.cut(raw["Physical Activity Level"], bins=[-np.inf, 54, 74, np.inf], labels=[0.0, 1.0, 2.0]).astype(float)
    raw["Physical Activity Encoded"]           = raw["Physical Activity Level Categorical"]
 
    # Standardize
    X = raw.copy()
    X[cols] = scaler.transform(X[cols])
 
    # Predictions
    prob = model.predict_proba(X)[0]
    y_pred, y_pred_set = mapie.predict_set(X, alpha=0.10)
    pred = int(y_pred[0])
    pclass = CLASS_LABELS[pred]
 
    pred_set = [CLASS_LABELS[i] for i, included in enumerate(y_pred_set[0]) if included]
    set_size = y_pred_set.sum(axis=1)
 
    # Results
    st.markdown(f"### Predicted Class:\n##### {pclass}")
 
    if bcc.predict(X)[0] == 1:
        st.markdown("###### It is advised that this patient case be sent for review.")
 
    st.divider()
 
    st.markdown(f"### Conformal Prediction Set (with a confidence level of 90%):\n##### {pred_set}")
    st.markdown("##### Interpretation: With a 90% confidence, we can say that the true label lies within the given prediction set.")
 
    if set_size == 1:
        st.markdown(f"##### The model is 90% confident that the patient has {pclass}.")
    else:
        st.markdown("###### The model is uncertain about the patient's diagnosis and it is advised that this case be sent for review.")
 
    st.divider()
 
    st.markdown(f"### Predicted Probabilities:")
    for cls, p in zip(CLASS_LABELS.values(), prob):
        st.markdown(f"##### {cls}: {p:.2f}")
    st.markdown(f"##### Interpretation: The patient has a predicted probability of {prob[pred]:.2f} of having {pclass}.")
 
    st.divider()
 
    # SHAP
    st.markdown("### SHapley Additive exPlanations (SHAP)")
    st.markdown("##### SHAP is a tool used for interpreting black box models.")
 
    shap_values = explainer(X)
 
    exp = shap.Explanation(
        values=shap_values.values[0, :, pred],
        base_values=shap_values.base_values[0, pred],
        data=X.iloc[0],
        feature_names=X.columns
    )
 
    shap.plots.waterfall(exp, max_display=8, show=False)
    plt.xlabel(f"Change in predicted probability of {pclass} (SHAP values)")
    st.pyplot(plt.gcf(), clear_figure=True)
 
    st.markdown(f"##### The standardized inputs are on the left for each feature. The base (expected) probability is {shap_values.base_values[0, pred]:.3f}.")
    st.markdown(
        f"""#### Interpretation for this Patient:
      - Gender:
      Affects the predicted probability of {pclass} by {shap_values.values[0, :, pred][0]:.2f} relative to the model's baseline probability.
 
      - Age:
      Affects the predicted probability of {pclass} by {shap_values.values[0, :, pred][1]:.2f} relative to the model's baseline probability.
 
      - Sleep Duration:
      Affects the predicted probability of {pclass} by {shap_values.values[0, :, pred][2]:.2f} relative to the model's baseline probability.
 
      - Quality of Sleep:
      Affects the predicted probability of {pclass} by {shap_values.values[0, :, pred][3]:.2f} relative to the model's baseline probability.
 
      - Physical Activity Level:
      Affects the predicted probability of {pclass} by {shap_values.values[0, :, pred][4]:.2f} relative to the model's baseline probability.
 
      - Stress Level:
      Affects the predicted probability of {pclass} by {shap_values.values[0, :, pred][5]:.2f} relative to the model's baseline probability.
 
      - Heart Rate:
      Affects the predicted probability of {pclass} by {shap_values.values[0, :, pred][6]:.2f} relative to the model's baseline probability.
 
      - Daily Steps:
      Affects the predicted probability of {pclass} by {shap_values.values[0, :, pred][7]:.2f} relative to the model's baseline probability.
        """
    )


    