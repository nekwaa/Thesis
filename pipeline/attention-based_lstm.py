# train_attention_lstm.py
import os
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D,
    TimeDistributed, Layer, Multiply, Permute, RepeatVector)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------------------------
# Helpers (column detection etc)
# ----------------------------
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
    co = find(["co", "carbon_monoxide"])
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
        try:
            rmse = mean_squared_error(y_true[:, i], y_pred[:, i], squared=False)
        except TypeError:
            rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{target} -> MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")


# ----------------------------
# Attention Layer (simple temporal attention)
# ----------------------------
class TemporalAttention(Layer):
    """
    Simple temporal attention layer.
    Input shape: (batch, timesteps, features)
    Output: (batch, features) — attention-weighted sum over timesteps
    """
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (batch, timesteps, features)
        self.timesteps = input_shape[1]
        self.features = input_shape[2]
        # Weight vector to compute attention scores for each timestep
        self.W = self.add_weight(name="attn_w", shape=(self.features,), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="attn_b", shape=(self.timesteps,), initializer="zeros", trainable=True)
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs shape: (batch, timesteps, features)
        # compute score for each timestep: score_t = tanh(sum(features * W) + b_t)
        # first compute f = inputs * W (elementwise across features) then sum over features
        # shape steps:
        #   inputs * W -> (batch, timesteps, features)
        #   sum over features -> (batch, timesteps)
        scores = tf.reduce_sum(inputs * self.W, axis=2) + self.b  # (batch, timesteps)
        scores = tf.nn.tanh(scores)
        # softmax over timesteps
        alphas = tf.nn.softmax(scores, axis=1)  # (batch, timesteps)
        # expand alphas to multiply with inputs
        alphas_expanded = tf.expand_dims(alphas, axis=2)  # (batch, timesteps, 1)
        weighted = inputs * alphas_expanded  # (batch, timesteps, features)
        output = tf.reduce_sum(weighted, axis=1)  # (batch, features)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


# ----------------------------
# Main training function
# ----------------------------
def main(csv_path, lookback=48, epochs=50, batch_size=64, model_dir="./models"):
    # load csv
    df = pd.read_csv(csv_path)

    # normalize column names (remove units/special chars, spaces -> underscore)
    df.columns = (
        df.columns
        .str.replace(r"[\(\)\%\?\µ\/]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.strip()
    )

    # parse timestamp if exists
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", dayfirst=True)
        df = df.sort_values("Timestamp").dropna(subset=["Timestamp"]).reset_index(drop=True)

    # drop non-numeric columns (we do not want strings in scalers)
    for col in df.select_dtypes(include="object").columns.tolist():
        # keep Timestamp if present (we handled it above)
        if col != "Timestamp":
            df.drop(columns=[col], inplace=True)

    # cast numeric & fill missing
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.interpolate(limit_direction="both").ffill().bfill()

    # clip sensible ranges (optional but helps stability)
    df = df.clip(lower=0)
    if "CO2_ppm" in df.columns:
        df["CO2_ppm"] = df["CO2_ppm"].clip(300, 2000)
    if "PM2.5_gm" in df.columns:
        df["PM2.5_gm"] = df["PM2.5_gm"].clip(0, 500)
    if "Temperature_C" in df.columns:
        df["Temperature_C"] = df["Temperature_C"].clip(-10, 50)
    if "Humidity_" in df.columns:
        df["Humidity_"] = df["Humidity_"].clip(0, 100)
    if "TVOC_ppb" in df.columns:
        df["TVOC_ppb"] = df["TVOC_ppb"].clip(0, 2000)

    # detect columns
    ts, co2, pm25, voc, temp, hum, co, light, occ = detect_columns(df)
    print("Detected columns:", ts, co2, pm25, voc, temp, hum, co, light, occ)

    # Compose feature and target lists (ensure consistent order)
    target_cols = [c for c in [co2, pm25, voc, temp, hum] if c]
    # optional extra features (append after targets so scaler ordering matches training plan)
    extra_features = [x for x in [co, light, occ] if x]
    feature_cols = target_cols + extra_features

    if len(target_cols) < 5:
        raise RuntimeError("Dataset must include CO2, PM2.5, VOC, Temperature, Humidity (or their equivalents)")

    print("Using feature columns:", feature_cols)
    print("Target columns:", target_cols)

    # build arrays
    df_features = df[feature_cols].astype(float)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(df_features.values)
    y_scaled = scaler_y.fit_transform(df[target_cols].values)

    # sequence creation
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback)
    if len(X_seq) == 0:
        raise RuntimeError("Not enough rows for given lookback. Reduce lookback or provide more data.")

    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Input shape: {X_train.shape} | Output shape: {y_train.shape}")

    # ----------------------------
    # Build Attention LSTM model
    # ----------------------------
    n_features = X_train.shape[2]
    inp = Input(shape=(lookback, n_features), name="input")
    # Small conv block to extract local patterns (optional, mirrors your CNN-LSTM)
    x = Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # LSTM stack (return_sequences=True for temporal attention)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)

    # attention: compute weighted sum over timesteps
    attn = TemporalAttention()(x)  # (batch, features)
    dense = Dense(64, activation="relu")(attn)
    dense = Dropout(0.2)(dense)
    out = Dense(len(target_cols), name="output")(dense)

    model = Model(inputs=inp, outputs=out, name="attention_lstm")
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    model.summary()

    # callbacks & training
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = os.path.join(model_dir, "attention_lstm_best.keras")
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

    # evaluate & metrics (invert scaling)
    y_pred = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    print("\n📊 Test Metrics (original units):")
    print_metrics(y_test_inv, y_pred_inv, target_cols)

    # save scalers + model
    joblib.dump(scaler_X, os.path.join(model_dir, "scaler_X_attention.save"))
    joblib.dump(scaler_y, os.path.join(model_dir, "scaler_y_attention.save"))
    model.save(os.path.join(model_dir, "attention_lstm_final.keras"))
    print(f"✅ Saved attention model + scalers to {model_dir}")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--lookback", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model_dir", type=str, default="./models")
    args = parser.parse_args()

    main(args.csv, lookback=args.lookback, epochs=args.epochs, batch_size=args.batch, model_dir=args.model_dir)
