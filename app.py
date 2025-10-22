import streamlit as st
import pandas as pd
import joblib
import folium
from streamlit_folium import st_folium
import time

CSV_FILE = "traffic_data_processed.csv"
MODEL_FILE = "rf_traffic_model.joblib"
FEATURES_FILE = "model_features.joblib"
DEFAULT_SIM_SPEED = 0.5  


df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
model = joblib.load(MODEL_FILE)
features = joblib.load(FEATURES_FILE)
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High"}

for f in features:
    if f not in df.columns:
        st.warning(f"Feature '{f}' not found in CSV. Filling with 0.")
        df[f] = 0


st.title("Real-Time Traffic Congestion Simulator with Map")
st.subheader("Predicted congestion levels at each station")

sim_speed = st.sidebar.slider(
    "Simulation Speed (seconds per row)", 0.1, 5.0, DEFAULT_SIM_SPEED, 0.1
)
start_idx = st.sidebar.number_input(
    "Start from row index", 0, len(df)-1, 0
)

table_placeholder = st.empty()
chart_placeholder = st.empty()
map_placeholder = st.empty()
predictions = []

st.write("Simulation running...")


for idx, row in df.iloc[start_idx:].iterrows():
   
    X = row[features].fillna(0).values.reshape(1, -1)
    pred = model.predict(X)[0]
    congestion_label = LABEL_MAP[pred]

    predictions.append({
        "timestamp": row["timestamp"],
        "station_name": row.get("station_name", "Unknown"),
        "lat": row.get("lat", 0),
        "lon": row.get("lon", 0),
        "predicted_congestion": congestion_label,
        "vehicle_count": row["vehicle_count"],
        "avg_speed": row["avg_speed"],
        "occupancy": row.get("occupancy", 0)
    })

  
    table_placeholder.table(pd.DataFrame(predictions[-20:]))

    chart_df = pd.DataFrame(predictions)
    chart_df["congestion_numeric"] = chart_df["predicted_congestion"].map(
        {"Low": 0, "Moderate": 1, "High": 2}
    )
    chart_placeholder.line_chart(
        chart_df.set_index("timestamp")["congestion_numeric"]
    )

   
    last_row = predictions[-1]
    fmap = folium.Map(location=[last_row["lat"], last_row["lon"]], zoom_start=12)

    for p in predictions[-20:]:
        folium.CircleMarker(
            location=[p["lat"], p["lon"]],
            radius=6,
            color={"Low":"green","Moderate":"orange","High":"red"}[p["predicted_congestion"]],
            fill=True,
            fill_opacity=0.7,
            popup=f"{p['station_name']}\nCongestion: {p['predicted_congestion']}\nVehicles: {p['vehicle_count']}"
        ).add_to(fmap)

    map_placeholder_folium = st_folium(fmap, width=700, height=400)

    time.sleep(sim_speed)
