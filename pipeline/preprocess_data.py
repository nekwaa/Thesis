# preprocess_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('data/IoT_Indoor_Air_Quality_Dataset.csv')

# Clean column names (remove units, spaces, etc.)
df.columns = df.columns.str.replace(r"[\(\)\%\?\µ\/]", "", regex=True)
df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

# Parse Timestamp as datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', dayfirst=True)

# Drop rows with missing timestamps or any other columns you find relevant
df = df.dropna(subset=['Timestamp'])

# Interpolate missing values for features
df = df.interpolate(limit_direction='both').ffill().bfill()

# Remove outliers using IQR method (for example CO2 levels)
Q1 = df['CO2_ppm'].quantile(0.25)
Q3 = df['CO2_ppm'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['CO2_ppm'] >= (Q1 - 1.5 * IQR)) & (df['CO2_ppm'] <= (Q3 + 1.5 * IQR))]

# Normalize features and target columns
features = ['CO2_ppm', 'PM2.5_gm', 'TVOC_ppb', 'Temperature_C', 'Humidity_']
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Scale features
X = df[features]
X_scaled = scaler_X.fit_transform(X)

# Save the scaled features back into the dataframe
df[features] = X_scaled

# Save the processed data to a new CSV file
df.to_csv('data/processed_air_quality_data.csv', index=False)

print("Preprocessing complete. Saved the processed data to 'processed_air_quality_data.csv'.")
