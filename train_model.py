import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("traffic_data_processed.csv", parse_dates=["timestamp"])
print(f"Loaded traffic_data_processed.csv with {len(df)} rows")

if "station_name" in df.columns:
    le_station = LabelEncoder()
    df["station_encoded"] = le_station.fit_transform(df["station_name"])
    joblib.dump(le_station, "station_encoder.joblib")  
    print("Encoded station_name into station_encoded")

FEATURES = ["vehicle_count", "avg_speed", "occupancy", "signal_time",
            "hour", "weekday", "veh_count_roll3", "avg_speed_roll3"]


if "station_encoded" in df.columns:
    FEATURES.append("station_encoded")

missing_feats = [f for f in FEATURES if f not in df.columns]
if missing_feats:
    print(f"ERROR: Missing features in CSV: {missing_feats}")
    exit(1)

X = df[FEATURES]
y = df["congestion_label"]

if X.isna().sum().sum() > 0:
    print("Warning: NaNs found in features. Filling with 0.")
    X = X.fillna(0)


split_idx = int(0.7 * len(df))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Training rows: {len(X_train)}, Testing rows: {len(X_test)}")

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")


pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nFeature Importances:")
for feat, imp in zip(FEATURES, model.feature_importances_):
    print(f"{feat}: {imp:.4f}")

joblib.dump(model, "rf_traffic_model.joblib")
joblib.dump(FEATURES, "model_features.joblib")
np.save("train_indices.npy", X_train.index.to_numpy())
np.save("test_indices.npy", X_test.index.to_numpy())

print("Model, features, and indices saved.")
