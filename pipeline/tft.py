# train_tft_fixed.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import json

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
)
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
    pm25 = find(["pm25", "pm2.5"])
    voc = find(["tvoc", "voc"])
    temp = find(["temp", "temperature"])
    hum = find(["humidity", "rh"])
    co = find(["co", "carbon_monoxide"])
    light = find(["light", "lux"])
    occ = find(["occupancy", "motion"])
    return ts, co2, pm25, voc, temp, hum, co, light, occ

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i + lookback])
        ys.append(y[i + lookback])
    return np.array(Xs), np.array(ys)

def print_metrics(y_true, y_pred, targets):
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{target} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

def tft_block(inputs, num_heads=4, ff_dim=128, dropout=0.2):
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attn_out = Dropout(dropout)(attn_out)
    attn_out = Add()([inputs, attn_out])
    attn_out = LayerNormalization()(attn_out)

    ff = Dense(ff_dim, activation='relu')(attn_out)
    ff = Dropout(dropout)(ff)
    ff_out = Dense(inputs.shape[-1])(ff)

    out = Add()([attn_out, ff_out])
    out = LayerNormalization()(out)
    return out

def main(csv_path, lookback=48, epochs=50, batch_size=64, model_dir="./models"):
    df = pd.read_csv(csv_path)

    df.columns = (
        df.columns
        .str.replace(r"[\(\)\%\?\µ\/]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip()
    )

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
        df = df.sort_values("Timestamp").dropna(subset=["Timestamp"]).reset_index(drop=True)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.interpolate(limit_direction="both").ffill().bfill()

    ts, co2, pm25, voc, temp, hum, co, light, occ = detect_columns(df)
    print("Detected columns:", ts, co2, pm25, voc, temp, hum, co, light, occ)

    target_cols = [c for c in [co2, pm25, voc, temp, hum] if c]
    extra_features = [x for x in [co, light, occ] if x]
    feature_cols = target_cols + extra_features

    if len(target_cols) < 5:
        raise RuntimeError("Dataset must include CO2, PM2.5, VOC, Temperature, Humidity")

    df_features = df[feature_cols].astype(float)
    X_values = df_features.values
    y_values = df[target_cols].values

    # ALWAYS scale
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_values)
    y_scaled = scaler_y.fit_transform(y_values)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback)
    if len(X_seq) == 0:
        raise RuntimeError("Not enough rows for given lookback.")
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    inp = Input(shape=(lookback, X_train.shape[2]))
    x = tft_block(inp)
    x = tft_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    lstm_out = LSTM(64, return_sequences=False)(inp)
    concat = tf.keras.layers.Concatenate()([x, lstm_out])
    dense = Dense(128, activation="relu")(concat)
    dense = Dropout(0.3)(dense)
    out = Dense(len(target_cols))(dense)

    model = Model(inputs=inp, outputs=out, name="TFT_Model")

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    model.summary()

    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "tft_best.keras")
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

    y_pred = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    print("\n📊 Test Metrics (original units):")
    print_metrics(y_test_inv, y_pred_inv, target_cols)

    joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X_tft.save"))
    joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y_tft.save"))
    model.save(os.path.join(model_dir, "tft_backup.keras"))
    print(f"✅ Model + scalers saved to {model_dir}")

    model_key = "tft"

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
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model_dir", type=str, default="./models")
    args = parser.parse_args()
    main(args.csv, lookback=args.lookback, epochs=args.epochs, batch_size=args.batch, model_dir=args.model_dir)
