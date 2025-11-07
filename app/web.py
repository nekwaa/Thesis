import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model

# --- Config ---
st.set_page_config(page_title="Air Quality Prediction System", layout="wide")
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_lstm_final.keras")
SCALER_X_PATH = os.path.join(MODEL_DIR, "scaler_X.save")
SCALER_Y_PATH = os.path.join(MODEL_DIR, "scaler_y.save")
LOOKBACK = 12

# --- Header ---
st.title("Air Quality Prediction System")
col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with col2:
    st.success("System Online")

# --- Load Model + Scalers ---
model, scaler_X, scaler_y = None, None, None
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH):
    model = load_model(MODEL_PATH, compile=False)
    scaler_X = joblib.load(SCALER_X_PATH)
    scaler_y = joblib.load(SCALER_Y_PATH)
    st.sidebar.success("✅ Model loaded successfully")
else:
    st.sidebar.error("⚠️ Model or scalers not found. Please train first.")

# --- Helpers ---
def detect_columns(df):
    cols = [c.lower() for c in df.columns]

    def find(candidates):
        for cand in candidates:
            for i, col in enumerate(cols):
                if cand in col:  # check substring match
                    return df.columns[i]
        return None

    ts = find(['timestamp', 'time', 'date', 'datetime'])
    co2 = find(['co2'])
    pm25 = find(['pm2.5', 'pm25'])
    voc = find(['tvoc', 'voc'])
    temp = find(['temp', 'temperature'])
    hum = find(['humidity', 'rh'])
    noise = find(['noise', 'decibel', 'sound'])

    return ts, co2, pm25, voc, temp, hum, noise


def multi_step_forecast(df, feature_cols, targets, ts_col, steps=[1,3,6,12,24]):
    """Multi-step forecasts with datetime labels"""
    df_features = df[feature_cols].astype(float).copy()
    df_features = df_features.interpolate(limit_direction='both').ffill().bfill()

    X_all = scaler_X.transform(df_features.values)
    if len(X_all) < LOOKBACK:
        return None

    X_last = X_all[-LOOKBACK:].copy()
    forecasts = {}

    # Time resolution
    if ts_col:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col])
            timestep = df[ts_col].diff().median()
        except Exception:
            timestep = timedelta(minutes=5)
    else:
        timestep = timedelta(minutes=5)

    last_time = df[ts_col].iloc[-1] if ts_col else datetime.now()

    for step in range(1, max(steps) + 1):
        X_input = np.expand_dims(X_last, axis=0)
        y_pred_scaled = model.predict(X_input, verbose=0)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0]

        if step in steps:
            forecasts[last_time + step * timestep] = dict(zip(targets, y_pred))

        # Insert predictions gradually to simulate dynamics
        new_row = np.copy(X_last[-1])
        for t_idx, target in enumerate(targets):
            if target in feature_cols:
                col_idx = feature_cols.index(target)
                new_row[col_idx] = y_pred_scaled[0, t_idx]
        X_last = np.vstack([X_last[1:], new_row])

    forecast_df = pd.DataFrame(forecasts).T
    forecast_df.index.name = "Forecast_Time"
    return forecast_df

# --- Data Input ---
st.subheader("Data Input")
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
data, forecast_df = None, None

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.dataframe(data, use_container_width=True)

    # Check if model is loaded correctly
    if model is None:
        st.error("Model is not loaded correctly.")
    if scaler_X is None or scaler_y is None:
        st.error("Scalers are not loaded correctly.")

    if model and scaler_X and scaler_y:
        ts, co2_col, pm25_col, voc_col, temp_col, hum_col, noise_col = detect_columns(data)
        # --- Match training features exactly ---
        # Model was trained on 8 features: CO2, PM2.5, VOC, Temp, Humidity + CO, Light, Occupancy
        extra_candidates = ['co', 'co_ppm', 'light', 'lux', 'occupancy', 'motion']

        extra_cols = []
        for cand in extra_candidates:
            for col in data.columns:
                if cand.lower() in col.lower():
                    extra_cols.append(col)

        # Remove duplicates and preserve order
        feature_cols = list(dict.fromkeys(
            [c for c in [co2_col, pm25_col, voc_col, temp_col, hum_col] if c is not None] + extra_cols
        ))

        # Keep only up to 8 features (the number model was trained on)
        feature_cols = feature_cols[:8]

        targets = [c for c in [co2_col, pm25_col, voc_col, temp_col, hum_col] if c is not None]


        if feature_cols and targets:
            forecast_df = multi_step_forecast(data, feature_cols, targets, ts)

# --- Forecast Display ---
st.subheader("Predictions")
if forecast_df is not None and not forecast_df.empty:
    # Debugging: check if forecast_df has data
    st.write(forecast_df)
    
    # Create 5 columns side by side
    cols = st.columns(len(forecast_df.columns))

    for i, col in enumerate(forecast_df.columns):
        with cols[i]:
            st.markdown(f"### {col}")
            # Show metrics horizontally
            for t, value in forecast_df[col].items():
                st.metric(f"{t.strftime('%Y-%m-%d %H:%M')}", f"{value:.2f}")
            # Show line chart per variable
            st.line_chart(forecast_df[[col]])
else:
    st.info("No prediction available")

# --- Noise (observed only) ---
st.subheader("Noise Levels (Observed)")
if data is not None and 'noise' in [c.lower() for c in data.columns]:
    st.line_chart(data[[noise_col]])
else:
    st.warning("No noise data in dataset")

# --- Alerts ---
st.subheader("Active Alerts")
if forecast_df is not None and not forecast_df.empty:
    for t, row in forecast_df.iterrows():
        if "PM2.5" in row and row["PM2.5"] > 25:
            st.warning(f"⚠️ PM₂.₅ > 25 µg/m³ at {t}: {row['PM2.5']:.1f}")
        if "CO2" in row and row["CO2"] > 1000:
            st.error(f"🚨 CO₂ > 1000 ppm at {t}: {row['CO2']:.1f}")
else:
    st.info("Upload data to see alerts")
