import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer


# ============================================================
#  Custom Temporal Attention Layer (needed for Attention LSTM)
# ============================================================
@tf.keras.utils.register_keras_serializable()
class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.timesteps = input_shape[1]
        self.features = input_shape[2]
        self.W = self.add_weight(
            name="attn_w",
            shape=(self.features,),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attn_b",
            shape=(self.timesteps,),
            initializer="zeros",
            trainable=True,
        )
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        scores = tf.reduce_sum(inputs * self.W, axis=2) + self.b
        alphas = tf.nn.softmax(tf.nn.tanh(scores), axis=1)
        weighted = inputs * tf.expand_dims(alphas, 2)
        return tf.reduce_sum(weighted, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


# ============================================================
#  Streamlit Config
# ============================================================
st.set_page_config(page_title="Air Quality Prediction System", layout="wide")
MODEL_DIR = "./models"
LOOKBACK = 96

# ============================================================
#  Header
# ============================================================
st.title("Air Quality Prediction System")

col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with col2:
    st.success("System Online")

# ============================================================
#  Model Selection
# ============================================================
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["CNN-LSTM", "Attention-based LSTM", "Temporal Fusion Transformer (TFT)"]
)

MODEL_MAP = {
    "CNN-LSTM": {
        "key": "cnn_lstm",
        "model": "cnn_lstm_best.keras",
        "scaler_X": "scaler_X_cnn_lstm.save",
        "scaler_y": "scaler_y_cnn_lstm.save",
        "metrics": "cnn_lstm_metrics.json",
        "meta": "cnn_lstm_meta.json",
        "n_features": 8,
    },

    "Attention-based LSTM": {
        "key": "attention_lstm",
        "model": "attention_lstm_best.keras",
        "scaler_X": "scaler_X_attention_lstm.save",
        "scaler_y": "scaler_y_attention_lstm.save",
        "metrics": "attention_lstm_metrics.json",
        "meta": "attention_lstm_meta.json",
        "n_features": 8,
    },

    "Temporal Fusion Transformer (TFT)": {
        "key": "tft",
        "model": "tft_best.keras",
        "scaler_X": "scaler_X_tft.save",
        "scaler_y": "scaler_y_tft.save",
        "metrics": "tft_metrics.json",
        "meta": "tft_meta.json",
        "n_features": 8,
    },
}

cfg = MODEL_MAP[model_choice]

model_path = os.path.join(MODEL_DIR, cfg["model"])
scaler_X_path = os.path.join(MODEL_DIR, cfg["scaler_X"])
scaler_y_path = os.path.join(MODEL_DIR, cfg["scaler_y"])
metrics_path = os.path.join(MODEL_DIR, cfg["metrics"])
meta_path = os.path.join(MODEL_DIR, cfg["meta"])

expected_features = cfg["n_features"]

# ============================================================
#  Load Model + Scalers
# ============================================================
model, scaler_X, scaler_y = None, None, None
custom_objects = {"TemporalAttention": TemporalAttention}

if os.path.exists(model_path) and os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
    try:
        model = load_model(model_path, compile=False, custom_objects=custom_objects)
        scaler_X = joblib.load(scaler_X_path)
        scaler_y = joblib.load(scaler_y_path)
        st.sidebar.success(f"{model_choice} loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Failed to load {model_choice}: {e}")
else:
    st.sidebar.error("⚠ Missing model or scaler files")


# ============================================================
#  Helpers
# ============================================================
def detect_columns(df):
    cols = [c.lower() for c in df.columns]

    def find(options):
        for opt in options:
            for i, col in enumerate(cols):
                if opt in col:
                    return df.columns[i]
        return None

    ts = find(["timestamp", "time", "date"])
    co2 = find(["co2"])
    pm25 = find(["pm25", "pm2.5"])
    voc = find(["voc", "tvoc"])
    temp = find(["temp"])
    hum = find(["humidity", "rh"])
    noise = find(["noise"])
    co = find(["co", "carbon_monoxide"])
    light = find(["light", "lux"])
    occ = find(["occupancy", "motion"])
    return ts, co2, pm25, voc, temp, hum, noise, co, light, occ


def multi_step_forecast(df, feature_cols, targets, ts_col, steps=[1, 3, 6, 12, 24]):
    df_features = df[feature_cols].astype(float).interpolate().ffill().bfill()
    X_all = scaler_X.transform(df_features.values)

    if len(X_all) < LOOKBACK:
        return None

    X_last = X_all[-LOOKBACK:]
    forecasts = {}

    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        timestep = df[ts_col].diff().median() or timedelta(minutes=5)
        last_time = df[ts_col].iloc[-1]
    else:
        timestep = timedelta(minutes=5)
        last_time = datetime.now()

    for step in range(1, max(steps) + 1):
        X_input = np.expand_dims(X_last, axis=0)

        y_scaled = model.predict(X_input, verbose=0)
        y_real = scaler_y.inverse_transform(y_scaled)[0]

        if step in steps:
            forecasts[last_time + step * timestep] = dict(zip(targets, y_real))

        new_row = np.copy(X_last[-1])
        for i, t in enumerate(targets):
            if t in feature_cols:
                idx = feature_cols.index(t)
                new_row[idx] = y_scaled[0, i]

        X_last = np.vstack([X_last[1:], new_row])

    df_fc = pd.DataFrame(forecasts).T
    df_fc.index.name = "Forecast_Time"
    return df_fc


# ============================================================
#  Data Upload
# ============================================================
st.subheader("Data Input")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
forecast_df = None

if uploaded:
    data = pd.read_csv(uploaded)
    st.dataframe(data, use_container_width=True)

    if model is None:
        st.error("Model/scalers not loaded")
    else:
        ts, co2, pm25, voc, temp, hum, noise, co, light, occ = detect_columns(data)
        feature_cols = [x for x in [co2, pm25, voc, temp, hum, co, light, occ] if x]
        targets = [x for x in [co2, pm25, voc, temp, hum] if x]

        # Adjust feature count
        if len(feature_cols) < expected_features:
            missing = expected_features - len(feature_cols)
            for i in range(missing):
                data[f"dummy_{i}"] = 0.0
                feature_cols.append(f"dummy_{i}")
        elif len(feature_cols) > expected_features:
            feature_cols = feature_cols[:expected_features]

        forecast_df = multi_step_forecast(data, feature_cols, targets, ts)

# ============================================================
#  Predictions
# ============================================================
st.subheader("Predictions")

if forecast_df is not None:
    forecast_df.columns = [c.replace(".", "_") for c in forecast_df.columns]
    st.dataframe(forecast_df, use_container_width=True)

    cols = st.columns(len(forecast_df.columns))
    for i, col in enumerate(forecast_df.columns):
        with cols[i]:
            st.markdown(f"### {col}")
            st.line_chart(forecast_df[[col]])
else:
    st.info("No predictions yet.")


# ============================================================
#  AQI COMPUTATION (PM2.5 + CO₂ Approximation)
# ============================================================
def compute_aqi_pm25(pm25):
    """
    Converts PM2.5 µg/m³ to AQI using USEPA breakpoints.
    """
    bp = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for (c_low, c_high, aqi_low, aqi_high) in bp:
        if c_low <= pm25 <= c_high:
            return ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low

    return None


def compute_aqi_co2(co2):
    """
    CO₂ is NOT part of official AQI.
    This produces a custom "Indoor CO₂ AQI".
    """
    if co2 < 800:
        return 50
    elif co2 < 1000:
        return 100
    elif co2 < 1500:
        return 150
    elif co2 < 2000:
        return 200
    elif co2 < 5000:
        return 300
    else:
        return 400

def aqi_category(aqi):
    if aqi <= 50:   return "Good", "green"
    if aqi <= 100:  return "Moderate", "yellow"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups", "orange"
    if aqi <= 200:  return "Unhealthy", "red"
    if aqi <= 300:  return "Very Unhealthy", "purple"
    return "Hazardous", "maroon"

# ============================================================
#  Alerts (AQI + PM2.5 + CO₂)
# ============================================================
st.subheader("Active Alerts")

if forecast_df is not None:

    pm25_col = None
    co2_col = None

    # Robust column matching
    for col in forecast_df.columns:
        c = col.lower()

        # Detect PM2.5
        if ("pm" in c) and ("2.5" in c or "2_5" in c or "25" in c):
            pm25_col = col

        # Detect CO2
        if "co2" in c:
            co2_col = col

    for t, row in forecast_df.iterrows():

        # --- PM2.5 AQI ---
        if pm25_col:
            pm25 = float(row[pm25_col])
            aqi = compute_aqi_pm25(pm25)

            if aqi is not None:
                category, color = aqi_category(aqi)

                if aqi > 100:
                    st.warning(
                        f"🌫 **PM2.5 AQI Alert** at {t}\n"
                        f"- PM2.5: {pm25:.1f} µg/m³\n"
                        f"- AQI: {aqi:.0f} ({category})"
                    )

        # --- CO₂ AQI ---
        if co2_col:
            co2 = float(row[co2_col])
            aqi = compute_aqi_co2(co2)

            if aqi is not None:
                category, color = aqi_category(aqi)

                if aqi > 100:
                    st.error(
                        f"🔥 **CO₂ Air Quality Alert** at {t}\n"
                        f"- CO₂: {co2:.0f} ppm\n"
                        f"- AQI Level: {aqi} ({category})"
                    )

# ============================================================
#  Model Performance Summary
# ============================================================
st.subheader("Model Performance Summary")

if os.path.exists(metrics_path) and os.path.exists(meta_path):

    import json

    # Load both JSON files
    metrics = json.load(open(metrics_path))
    meta = json.load(open(meta_path))

    # Extract Metric Values
    mae_vals = metrics.get("mae", {})
    accuracy_vals = metrics.get("accuracy", {})
    avg_accuracy = np.mean(list(accuracy_vals.values())) if accuracy_vals else 0

    # Confidence rating
    confidence = (
        "High" if avg_accuracy > 85 else
        "Medium" if avg_accuracy > 60 else
        "Low"
    )

    # Extract Meta Values
    last_trained = meta.get("last_trained", "N/A")
    update_freq = meta.get("update_frequency", "N/A")
    data_points = meta.get("data_points", "N/A")

    # Two-column layout
    left, right = st.columns([1, 1])

    # LEFT — Model Information
    with left:
        st.markdown("""
        <div style="padding:20px;border-radius:12px;background-color:#44444E;">
        <h4>Model Information</h4>
        """, unsafe_allow_html=True)

        st.write("**Model Type:**", model_choice)
        st.write("**Last Training:**", last_trained)
        st.write(f"**Accuracy Score:** {avg_accuracy:.1f}%")
        st.write("**Confidence Level:**", confidence)

        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT — Metrics
    with right:
        st.markdown("""
        <div style="padding:20px;border-radius:12px;background-color:#44444E;">
        <h4>Summary Metrics</h4>
        """, unsafe_allow_html=True)

        st.write("**Mean Absolute Error (per variable):**")
        for k, v in mae_vals.items():
            st.write(f"- **{k}:** {v:.2f}")

        st.write("**Prediction Window:** 6 hours")
        st.write("**Data Points:**", f"{data_points:,}")
        st.write("**Update Frequency:**", update_freq)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("No model metrics available. Train a model first.")