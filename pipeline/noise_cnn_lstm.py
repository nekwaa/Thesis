# train_noise_cnn_lstm.py
import os
import argparse
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

RNG_SEED = 42
tf.random.set_seed(RNG_SEED)
np.random.seed(RNG_SEED)

def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    def find(cands):
        for cand in cands:
            for i, c in enumerate(cols):
                if cand in c:
                    return df.columns[i]
        return None
    ts = find(["timestamp", "time", "date", "datetime"])
    noise = find(["noise", "db", "decibel"])
    # optional extras (if present) but we'll treat them as features
    temp = find(["temp", "temperature"])
    occ = find(["occupancy", "occupation", "engagement", "engagement_score"])
    feedback = find(["feedback", "response", "reaction"])
    return ts, noise, temp, occ, feedback

def create_sequences(X, y, lookback):
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])
    return np.array(Xs), np.array(ys)

def print_metrics(y_true, y_pred, target_name="noise"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    acc = max(0, 100 - mape)
    print(f"{target_name} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}, MAPE: {mape:.2f}%, Accuracy: {acc:.2f}%")
    return mae, rmse, mape, acc, r2

def main(csv_path, lookback=96, epochs=50, batch_size=64, model_dir="./models_noise"):
    df = pd.read_csv(csv_path, sep=None, engine="python")
    # normalize column names
    df.columns = (
        df.columns
        .str.replace(r"[\(\)\%\?\µ\/]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip()
    )

    ts_col, noise_col, temp_col, occ_col, feedback_col = detect_columns(df)
    print("Detected columns:", ts_col, noise_col, temp_col, occ_col, feedback_col)

    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", dayfirst=True)
        df = df.sort_values(ts_col).dropna(subset=[ts_col]).reset_index(drop=True)

    # drop non-numeric columns (keep timestamp)
    for col in df.select_dtypes(include="object").columns.tolist():
        if col != ts_col:
            df.drop(columns=[col], inplace=True)

    # convert numeric and interpolate
    df = df.apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").ffill().bfill()

    if noise_col is None:
        raise RuntimeError("Could not detect any noise column (e.g., 'noise', 'db', 'classroom_noise') in CSV")

    # sensible clipping for dB
    df[noise_col] = df[noise_col].clip(lower=0, upper=140)

    # features: use numeric columns except the target
    feature_cols = [c for c in df.columns if c != noise_col and c != ts_col]
    if len(feature_cols) == 0:
        # no extra features: create a dummy feature so the model still receives 1 input
        df["dummy"] = 0.0
        feature_cols = ["dummy"]

    X_values = df[feature_cols].values.astype(float)
    y_values = df[noise_col].values.astype(float).reshape(-1, 1)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_values)
    y_scaled = scaler_y.fit_transform(y_values)

    X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback)
    if len(X_seq) == 0:
        raise RuntimeError("Not enough rows for given lookback. Provide more data or reduce lookback.")

    split = int(0.7 * len(X_seq))  # thesis uses 70/30
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape} | Output shape: {y_train.shape}")

    # Model
    model = Sequential([
        Input(shape=(lookback, X_train.shape[2])),
        Conv1D(64, 3, activation="relu", padding="causal"),
        BatchNormalization(),
        Conv1D(32, 3, activation="relu", padding="causal"),
        MaxPooling1D(2),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1)  # single target
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])

    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "cnn_lstm_noise_best.keras")
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_loss", mode="min")
    earlystop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs, batch_size=batch_size,
        callbacks=[checkpoint, earlystop],
        verbose=1
    )

    y_pred = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    print("\n📊 Test Metrics (original units):")
    mae, rmse, mape, acc, r2 = print_metrics(y_test_inv.ravel(), y_pred_inv.ravel(), "Classroom_Noise")

    # save scalers, model, metrics, metadata
    joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X_cnn_lstm_noise.save"))
    joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y_cnn_lstm_noise.save"))
    model.save(os.path.join(model_dir, "cnn_lstm_noise_backup.keras"))

    metrics_json = {
        "mae": {"Classroom_Noise": float(mae)},
        "rmse": {"Classroom_Noise": float(rmse)},
        "mape": {"Classroom_Noise": float(mape)},
        "accuracy": {"Classroom_Noise": float(acc)},
        "r2": {"Classroom_Noise": float(r2)}
    }
    with open(os.path.join(model_dir, "cnn_lstm_noise_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=4)

    metadata = {
        "model_name": "cnn_lstm_noise",
        "last_trained": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "data_points": int(len(df)),
        "lookback": int(lookback),
        "update_frequency": "1 min"
    }
    with open(os.path.join(model_dir, "cnn_lstm_noise_meta.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Model + scalers saved to {model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model_dir", type=str, default="./models_noise")
    args = parser.parse_args()
    main(args.csv, lookback=args.lookback, epochs=args.epochs, batch_size=args.batch, model_dir=args.model_dir)
