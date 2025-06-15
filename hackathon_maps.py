
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import geocoder


# Function to preprocess the dataset
def preprocess_dataset(df):
    # Drop rows with missing latitude or longitude
    df = df.dropna(subset=["lattitude", "longitude"])
    # Ensure latitude and longitude are numeric
    df["lattitude"] = pd.to_numeric(df["lattitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    # Drop rows with invalid latitude or longitude
    df = df.dropna(subset=["lattitude", "longitude"])
    return df

# Function to find the nearest charging station
def find_nearest_station(user_location, stations_df):
    stations_df["distance_km"] = stations_df.apply(
        lambda row: geodesic(user_location, (row["lattitude"], row["longitude"])).kilometers, axis=1
    )
    nearest_station = stations_df.loc[stations_df["distance_km"].idxmin()]
    return nearest_station

# Streamlit app
st.title("EV Charging Station Finder")

# Load the dataset from the project directory
DATASET_PATH = "ev-charging-stations-india.csv"

try:
    stations_df = pd.read_csv(DATASET_PATH)

    # Validate the dataset
    required_columns = ["name", "state", "city", "address", "lattitude", "longitude", "type"]
    if not all(col in stations_df.columns for col in required_columns):
        st.error(f"Dataset must contain the following columns: {', '.join(required_columns)}")
    else:
        st.success("Dataset loaded successfully!")

        # Preprocess the dataset
        stations_df = preprocess_dataset(stations_df)

        # Get user's current location
        g = geocoder.ip("me")
        if g.ok:
            user_location = g.latlng

            st.write("### Your Current Location")
            st.write(f"Latitude: {user_location[0]}, Longitude: {user_location[1]}")

            # Find the nearest EV charging station
            nearest_station = find_nearest_station(user_location, stations_df)

            st.write("### Nearest EV Charging Station")
            st.write(f"*Name:* {nearest_station['name']}")
            st.write(f"*Address:* {nearest_station['address']}, {nearest_station['city']}, {nearest_station['state']}")
            st.write(f"*Type:* {nearest_station['type']}")
            st.write(f"*Distance:* {nearest_station['distance_km']:.2f} km")

            # Display locations on map
            st.write("### Map")
            m = folium.Map(location=user_location, zoom_start=12)

            # Add user's location to the map
            folium.Marker(user_location, popup="Your Location", icon=folium.Icon(color="blue")).add_to(m)

            # Add nearest station to the map
            station_location = (nearest_station["lattitude"], nearest_station["longitude"])
            folium.Marker(
                station_location,
                popup=f"{nearest_station['name']}\n{nearest_station['address']}",
                icon=folium.Icon(color="green"),
            ).add_to(m)

            # Render map in Streamlit
            st_folium(m, width=700, height=500)
        else:
            st.error("Unable to fetch your current location. Please ensure location services are enabled.")
except FileNotFoundError:
    st.error("Dataset file not found in the project directory. Please ensure 'ev_charging_stations.csv' exists.")
