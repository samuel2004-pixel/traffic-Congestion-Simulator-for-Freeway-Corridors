import os
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
from sklearn.preprocessing import LabelEncoder

print("Starting preprocessing...")


CSV_FILE = "traffic_freeway_corridor_100.csv"
OUTPUT_FILE = "traffic_data_processed.csv"
REQUIRED_COLS = ["timestamp", "vehicle_count", "avg_speed", "congestion_level", "station_name", "lat", "lon"]


print("Current folder:", os.getcwd())
if not os.path.exists(CSV_FILE):
    print(f"ERROR: {CSV_FILE} not found in {os.getcwd()}")
    exit(1)

try:
    df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
    print(f"Loaded {CSV_FILE} with {len(df)} rows")
except FileNotFoundError:
    print(f"ERROR: {CSV_FILE} not found")
    exit(1)
except EmptyDataError:
    print(f"ERROR: {CSV_FILE} is empty")
    exit(1)
except ParserError as e:
    print(f"ERROR parsing CSV: {e}")
    exit(1)

missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
if missing_cols:
    print(f"ERROR: Missing columns in CSV: {missing_cols}")
    exit(1)


print("First 5 rows:\n", df.head())


df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["weekday"] = df["timestamp"].dt.weekday
df["time_of_day"] = df["hour"] + df["minute"] / 60.0


df["veh_count_roll3"] = df["vehicle_count"].rolling(3, min_periods=1).mean()
df["avg_speed_roll3"] = df["avg_speed"].rolling(3, min_periods=1).mean()

le = LabelEncoder()
df["congestion_label"] = le.fit_transform(df["congestion_level"])


df = df.dropna().reset_index(drop=True)
print(f"After dropping NaNs: {len(df)} rows")


df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {OUTPUT_FILE} with {len(df)} rows")
