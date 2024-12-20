import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import geocoder
from plyer import notification
import pickle
from sklearn.preprocessing import StandardScaler 
#Page Configuration
st.set_page_config(page_title="EV Charging Station Finder", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
        .title {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 36px;
            color: #4C9EFF;
            font-weight: bold;
            text-align: center;
        }
        .header {
            font-family: 'Roboto', sans-serif;
            font-size: 28px;
            color: #333;
            font-weight: 600;
        }
        .subheader {
            font-family: 'Roboto', sans-serif;
            font-size: 22px;
            color: #555;
        }
        .card {
            background-color: #f4f4f4;
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to preprocess the dataset
def preprocess_dataset(df):
    df = df.dropna(subset=["lattitude", "longitude"])
    df["lattitude"] = pd.to_numeric(df["lattitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["lattitude", "longitude"])
    return df

# Function to validate range of input
def validate_input(value, min_val, max_val):
    try:
        value = float(value)
        if min_val <= value <= max_val:
            return value, None
        else:
            return None, f"Value {value} is out of range! Expected: {min_val} - {max_val}"
    except ValueError:
        return None, "Invalid input! Please enter a number."

# Function to find the nearest charging station
def find_nearest_station(user_location, stations_df):
    stations_df["distance_km"] = stations_df.apply(
        lambda row: geodesic(user_location, (row["lattitude"], row["longitude"])).kilometers, axis=1
    )
    nearest_station = stations_df.loc[stations_df["distance_km"].idxmin()]
    return nearest_station, stations_df.nsmallest(5, "distance_km")

# Sidebar Section
with st.sidebar:
    st.image("https://i.imgur.com/vnIHKPf.png", width=200)
    st.title("EV Finder ⚡")
    st.markdown(
        """
        **Find the nearest EV charging station** with just a few clicks.
        - 📍 Nearest station map view
        - ⚠️ Battery alert notifications
        """
    )

# Main Content Area
st.markdown('<div class="title">EV Charging Station Finder</div>', unsafe_allow_html=True)

# Add Input Fields with Ranges
st.subheader("🔢 Input Vehicle Data")

fields = {
    "Time [s]": (0.0, 3164.4),
    "Velocity [km/h]": (0.0, 143.82),
    "Elevation [m]": (479.0, 664.99),
    "Throttle [%]": (0.0, 99.63),
    "Motor Torque [Nm]": (-87.9, 249.5),
    "Battery Voltage [V]": (349.41, 394.66),
    "Battery Current [A]": (-395.18, 143.53),
    "Battery Temperature [Â°C]": (16.0, 32.0),
    "max. SoC [%)": (89.5, 90.0),
    "Requested Heating Power [W]": (0.0, 7000.0),
    "AirCon Power [kW]": (0.0, 3.32),
    "Ambient Temperature [Â°C]": (14.0, 33.5),
    "Heat Exchanger Temperature [Â°C]": (5.0, 50.0),
}

inputs = {}
errors = []

pickle_in = open("rf_model.pkl","rb")

model = pickle.load(pickle_in)
scaler_x = StandardScaler()
scaler_y = StandardScaler()
with open("scaler_x.pkl", "rb") as f:
    scaler_x = pickle.load(f)

with open("scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

for field, (min_val, max_val) in fields.items():
    # Set default values for specific fields
    default_values = {
        "max. SoC [%)": "90",  # Pre-filled with 90
        "Requested Heating Power [W]": "5000",  # Example default value
        "AirCon Power [kW]": "1.5",  # Example default value
        "Ambient Temperature [Â°C]": "25",  # Example default value
        "Heat Exchanger Temperature [Â°C]": "30",  # Example default value
    }
    
    # Use the default value if it exists; otherwise, leave it blank
    default_value = default_values.get(field, " ")
    
    # Render text input with default value
    user_input = st.text_input(f"{field} (Range: {min_val} - {max_val})", default_value)
    value, error = validate_input(user_input, min_val, max_val)
    
    if error:
        errors.append(f"{field}: {error}")
    inputs[field] = value


if errors:
    st.error("\n".join(errors))
else:
    if st.button("Calculate SoC"):
        manual_df = pd.DataFrame([inputs])
        manual_input_scaled = scaler_x.transform(manual_df)
        manual_pred_scaled = model.predict(manual_input_scaled)

    # Reshape manual_pred_scaled to a 2D array
        manual_pred_scaled = manual_pred_scaled.reshape(-1, 1)  

        soc = scaler_y.inverse_transform(manual_pred_scaled)
       
        st.success(f"Calculated SoC: {soc} %")
# After calculating the SoC
if errors:
    st.error("\n".join(errors))
else:
    if st.button("Calculate SoC"):
        manual_df = pd.DataFrame([inputs])
        manual_input_scaled = scaler_x.transform(manual_df)
        manual_pred_scaled = model.predict(manual_input_scaled)

        # Reshape manual_pred_scaled to a 2D array
        manual_pred_scaled = manual_pred_scaled.reshape(-1, 1)

        soc = scaler_y.inverse_transform(manual_pred_scaled)
        st.success(f"Calculated SoC: {soc[0][0]:.2f} %")

        # Check if SoC is less than 45
        if soc[0][0] < 45:
            st.warning("Warning: SoC is less than 45%. Finding the nearest charging station...")

            # Add JavaScript to scroll to the map section
            st.markdown(
                """
                <script>
                    document.querySelector('section[data-testid="stVerticalBlock"]')
                        .scrollIntoView({behavior: 'smooth'});
                </script>
                """,
                unsafe_allow_html=True,
            )
# Dataset Path
DATASET_PATH = "ev-charging-stations-india.csv"

try:
    stations_df = pd.read_csv(DATASET_PATH)
    required_columns = ["name", "state", "city", "address", "lattitude", "longitude", "type"]

    if not all(col in stations_df.columns for col in required_columns):
        st.error(f"Dataset must contain columns: {', '.join(required_columns)}")
    else:
        stations_df = preprocess_dataset(stations_df)

        # Fetch user's location
        g = geocoder.ip("me")
        if g.ok:
            user_location = g.latlng

            st.subheader("📍 Nearest Charging Station")
            nearest_station, top_stations = find_nearest_station(user_location, stations_df)

            st.markdown(f"<div class='header'>{nearest_station['name']}</div>", unsafe_allow_html=True)
            st.write(f"**Address:** {nearest_station['address']}, {nearest_station['city']}")
            st.write(f"**Type:** {nearest_station['type']}")
            st.write(f"**Distance:** {nearest_station['distance_km']:.2f} km")

            # Display map
            m = folium.Map(location=user_location, zoom_start=12)
            folium.Marker(user_location, popup="You are here!", icon=folium.Icon(color="blue")).add_to(m)
            station_location = (nearest_station["lattitude"], nearest_station["longitude"])
            folium.Marker(
                station_location,
                popup=nearest_station["name"],
                icon=folium.Icon(color="green"),
            ).add_to(m)
            st_folium(m, width=700, height=400)

            # Display Top 5 Stations
            st.subheader("🏆 Top 5 Nearest Charging Stations")
            st.dataframe(
                top_stations[["name", "address", "city", "distance_km"]]
                .rename(columns={"distance_km": "Distance (km)"})
            )

except FileNotFoundError:
    st.error("Dataset file not found! Make sure 'ev-charging-stations-india.csv' exists.")

