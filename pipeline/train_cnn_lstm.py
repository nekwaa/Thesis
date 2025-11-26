# train_cnn_lstm_fixed.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

RNG_SEED = 42
tf.random.set_seed(RNG_SEED)
np.random.seed(RNG_SEED)

def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    def find(candidates):
        for cand in candidates:
            for i, col in enumerate(cols):
                if cand in col:
                    return df.columns[i]
        return None

    ts = find(["timestamp", "time", "date", "datetime"])
    co2 = find(["co2"])
    pm25 = find(["pm2.5", "pm25"])
    voc = find(["tvoc", "voc"])
    temp = find(["temp", "temperature"])
    hum = find(["humidity", "rh"])
    co = find(["carbon_monoxide", "co_ppm", "co (ppm)", "co"])
    light = find(["light", "lux"])
    occ = find(["occupancy", "motion"])
    return ts, co2, pm25, voc, temp, hum, co, light, occ

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])
    return np.array(Xs), np.array(ys)

def print_metrics(y_true, y_pred, targets):
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{target} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

def main(csv_path, lookback=48, epochs=50, batch_size=64, model_dir="./models"):
    df = pd.read_csv(csv_path)

    # --- Clean column names ---
    df.columns = (
        df.columns
        .str.replace(r"[\(\)\%\?\µ\/]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip()
    )

    # --- Parse timestamps ---
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
        df = df.sort_values("Timestamp").dropna(subset=["Timestamp"]).reset_index(drop=True)

    # --- Drop non-numeric categorical columns (keep Timestamp if present) ---
    for col in df.select_dtypes(include="object").columns.tolist():
        if col != "Timestamp":
            df.drop(columns=[col], inplace=True)

    # --- Convert to numeric and fill missing ---
    df = df.apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").ffill().bfill()

    # --- Clip sensible ranges ---
    df = df.clip(lower=0)
    if "CO2_ppm" in df.columns:
        df["CO2_ppm"] = df["CO2_ppm"].clip(300, 2000)
    if "PM2.5_gm" in df.columns:
        df["PM2.5_gm"] = df["PM2.5_gm"].clip(0, 500)
    if "Temperature_C" in df.columns:
        df["Temperature_C"] = df["Temperature_C"].clip(-20, 60)
    if "Humidity_" in df.columns:
        df["Humidity_"] = df["Humidity_"].clip(0, 100)
    if "TVOC_ppb" in df.columns:
        df["TVOC_ppb"] = df["TVOC_ppb"].clip(0, 2000)

    # --- Detect columns ---
    ts, co2, pm25, voc, temp, hum, co, light, occ = detect_columns(df)
    print("Detected columns:", ts, co2, pm25, voc, temp, hum, co, light, occ)

    target_cols = [c for c in [co2, pm25, voc, temp, hum] if c]
    extra_features = [x for x in [co, light, occ] if x]
    feature_cols = target_cols + extra_features

    if len(target_cols) < 5:
        raise RuntimeError("Dataset must include CO2, PM2.5, VOC, Temperature, Humidity")

    print(f"✅ Features used: {feature_cols}")
    print(f"🎯 Targets: {target_cols}")

    df_features = df[feature_cols].astype(float)
    X_values = df_features.values
    y_values = df[target_cols].values

    # --- ALWAYS apply scalers (do not auto-skip) ---
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_values)
    y_scaled = scaler_y.fit_transform(y_values)

    # --- Sequences & split ---
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback)
    if len(X_seq) == 0:
        raise RuntimeError("Not enough rows for given lookback. Provide more data or reduce lookback.")
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape} | Output shape: {y_train.shape}")

    # --- Model ---
    model = Sequential([
        Input(shape=(lookback, len(feature_cols))),
        Conv1D(64, 3, activation="relu", padding="causal"),
        BatchNormalization(),
        Conv1D(32, 3, activation="relu", padding="causal"),
        MaxPooling1D(2),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(len(target_cols))
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])

    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "cnn_lstm_best.keras")
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", mode="min")
    earlystop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, earlystop],
        verbose=1
    )

    # --- Evaluate & inverse scale ---
    y_pred = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    print("\n📊 Test Metrics (original units):")
    print_metrics(y_test_inv, y_pred_inv, target_cols)

    # --- Save scalers + backup model ---
    joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X_cnn_lstm.save"))
    joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y_cnn_lstm.save"))
    model.save(os.path.join(model_dir, "cnn_lstm_backup.keras"))
    print(f"✅ Model + scalers saved to {model_dir}")


    model_key = "cnn_lstm"

    # ---- Save metrics JSON ----
    metrics_json = {"mae": {}, "rmse": {}, "mape": {}, "accuracy": {}, "r2": {}}

    for i, col in enumerate(target_cols):
        y_t = y_test_inv[:, i]
        y_p = y_pred_inv[:, i]

        mae = float(mean_absolute_error(y_t, y_p))
        rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
        r2 = float(r2_score(y_t, y_p))

        # avoid division by zero
        mape = float(np.mean(np.abs((y_t - y_p) / np.clip(np.abs(y_t), 1e-6, None))) * 100)
        accuracy = float(max(0, 100 - mape))

        metrics_json["mae"][col] = mae
        metrics_json["rmse"][col] = rmse
        metrics_json["mape"][col] = mape
        metrics_json["accuracy"][col] = accuracy
        metrics_json["r2"][col] = r2

    with open(os.path.join(model_dir, f"{model_key}_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=4)
        
    # ---- Save metadata JSON ----
    metadata = {
        "model_name": model_key,
        "last_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data_points": len(df),
        "lookback": lookback,
        "update_frequency": "5 min",
    }

    with open(os.path.join(model_dir, f"{model_key}_meta.json"), "w") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model_dir", type=str, default="./models")
    args = parser.parse_args()
    main(args.csv, lookback=args.lookback, epochs=args.epochs, batch_size=args.batch, model_dir=args.model_dir)
