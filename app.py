
import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.drone import Drone
from core.physics import calculate_acceleration
from core.controller import PIDController

from mapping.map_engine import (
    download_buildings,
    load_geojson,
    load_elevation_csv,
)

from navigation.planner import (
    route_distance,
    check_building_intersections,
)

from ai.trainer import train_agent

from visualization.charts import (
    flight_chart,
    reward_chart,
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AeroMind",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚁 AeroMind")
st.caption(
    "Autonomous UAV Navigation & Digital Twin Platform"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}
div[data-testid="stMetric"] {
    background-color: rgba(128,128,128,0.08);
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# SESSION STATE
# =====================================================

DEFAULTS = {
    "buildings": [],
    "elevation": None,
    "flight_data": None,
    "q_table": None,
    "training_rewards": None,
    "mission_waypoints": [],
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def create_download_buffer(array):
    """Convert NumPy array into downloadable binary data."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)
    return buffer.getvalue()


def make_trajectory_figure(df):
    """Create 3D trajectory from simulation data."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=df["x"],
            y=df["y"],
            z=df["altitude"],
            mode="lines",
            name="Drone trajectory",
            line=dict(width=5),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[df["x"].iloc[0]],
            y=[df["y"].iloc[0]],
            z=[df["altitude"].iloc[0]],
            mode="markers",
            name="Start",
            marker=dict(size=7),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[df["x"].iloc[-1]],
            y=[df["y"].iloc[-1]],
            z=[df["altitude"].iloc[-1]],
            mode="markers",
            name="End",
            marker=dict(size=7),
        )
    )

    fig.update_layout(
        title="3D Drone Flight Trajectory",
        scene=dict(
            xaxis_title="X position (m)",
            yaxis_title="Y position (m)",
            zaxis_title="Altitude (m)",
        ),
        height=500,
    )

    return fig


def display_error(error):
    st.error(f"{type(error).__name__}: {error}")


# =====================================================
# SIDEBAR - DRONE CONFIGURATION
# =====================================================

st.sidebar.title("Drone Configuration")

mass = st.sidebar.number_input(
    "Drone mass (kg)",
    min_value=0.5,
    max_value=20.0,
    value=3.0,
    step=0.1,
)

payload = st.sidebar.number_input(
    "Payload (kg)",
    min_value=0.0,
    max_value=10.0,
    value=0.5,
    step=0.1,
)

max_thrust = st.sidebar.number_input(
    "Maximum total thrust (N)",
    min_value=5.0,
    max_value=300.0,
    value=60.0,
    step=1.0,
)

battery = st.sidebar.slider(
    "Initial battery (%)",
    min_value=1,
    max_value=100,
    value=100,
)

wind_speed = st.sidebar.slider(
    "Wind speed (m/s)",
    min_value=0.0,
    max_value=25.0,
    value=3.0,
)

wind_direction = st.sidebar.slider(
    "Wind direction (degrees)",
    min_value=0,
    max_value=359,
    value=90,
)

total_mass = mass + payload

st.sidebar.divider()

st.sidebar.metric(
    "Total mass",
    f"{total_mass:.2f} kg",
)

st.sidebar.metric(
    "Weight",
    f"{total_mass * 9.81:.2f} N",
)


# =====================================================
# MAIN NAVIGATION TABS
# =====================================================

tabs = st.tabs([
    "Mission Control",
    "Map Intelligence",
    "Drone Simulation",
    "AI Training",
    "Analytics",
])


# =====================================================
# TAB 1 - MISSION CONTROL
# =====================================================

with tabs[0]:

    st.subheader("Mission Control")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total mass",
        f"{total_mass:.2f} kg",
    )

    c2.metric(
        "Battery",
        f"{battery}%",
    )

    c3.metric(
        "Wind speed",
        f"{wind_speed:.1f} m/s",
    )

    st.divider()

    st.markdown("### Mission Coordinates")

    col1, col2 = st.columns(2)

    with col1:

        start_lat = st.number_input(
            "Starting latitude",
            value=27.1767,
            format="%.6f",
        )

        start_lon = st.number_input(
            "Starting longitude",
            value=78.0081,
            format="%.6f",
        )

    with col2:

        end_lat = st.number_input(
            "Destination latitude",
            value=27.1800,
            format="%.6f",
        )

        end_lon = st.number_input(
            "Destination longitude",
            value=78.0150,
            format="%.6f",
        )

    altitude = st.number_input(
        "Target altitude above ground (m)",
        min_value=2.0,
        max_value=120.0,
        value=30.0,
    )

    waypoints = [
        (start_lat, start_lon),
        (end_lat, end_lon),
    ]

    st.session_state.mission_waypoints = waypoints

    distance = route_distance(waypoints)

    st.metric(
        "Straight-line distance",
        f"{distance:.2f} m",
    )

    st.caption(
        "Geographic distance is calculated using "
        "the Haversine formula."
    )

    if st.session_state.buildings:

        try:

            collisions = check_building_intersections(
                waypoints,
                st.session_state.buildings,
            )

            if collisions:

                st.warning(
                    f"{len(collisions)} potential building "
                    "intersections detected."
                )

            else:

                st.success(
                    "No intersections detected in loaded building data."
                )

        except Exception as error:
            display_error(error)

    else:

        st.info(
            "Load building footprints in Map Intelligence "
            "to perform a basic route intersection check."
        )


# =====================================================
# TAB 2 - MAP INTELLIGENCE
# =====================================================

with tabs[1]:

    st.subheader("Geospatial Intelligence")

    st.markdown("### OpenStreetMap Building Retrieval")

    col1, col2 = st.columns(2)

    with col1:

        map_lat = st.number_input(
            "Map center latitude",
            value=27.1767,
            format="%.6f",
            key="map_lat",
        )

    with col2:

        map_lon = st.number_input(
            "Map center longitude",
            value=78.0081,
            format="%.6f",
            key="map_lon",
        )

    radius = st.slider(
        "Search radius (metres)",
        min_value=100,
        max_value=1500,
        value=500,
    )

    if st.button(
        "Download building data",
        key="download_buildings",
    ):

        try:

            with st.spinner(
                "Downloading OpenStreetMap data..."
            ):

                buildings = download_buildings(
                    map_lat,
                    map_lon,
                    radius,
                )

                st.session_state.buildings = buildings

            st.success(
                f"{len(buildings)} building features retrieved."
            )

        except Exception as error:
            display_error(error)

    st.divider()

    st.markdown("### Upload GeoJSON")

    geo_file = st.file_uploader(
        "Upload building footprints or geographic features",
        type=["geojson", "json"],
        key="geojson_upload",
    )

    if st.button(
        "Process GeoJSON",
        disabled=geo_file is None,
    ):

        try:

            geo_file.seek(0)
            features = load_geojson(geo_file)

            new_buildings = []

            for feature in features:

                geom = feature["geometry"]

                if geom.geom_type == "Polygon":

                    coords = list(geom.exterior.coords)

                elif geom.geom_type == "MultiPolygon":

                    largest = max(
                        geom.geoms,
                        key=lambda polygon: polygon.area,
                    )

                    coords = list(largest.exterior.coords)

                else:
                    continue

                center = geom.centroid

                new_buildings.append({
                    "geometry": coords,
                    "lat": center.y,
                    "lon": center.x,
                })

            st.session_state.buildings.extend(
                new_buildings
            )

            st.success(
                f"{len(new_buildings)} building geometries processed."
            )

        except Exception as error:
            display_error(error)

    st.divider()

    st.markdown("### Elevation Data")

    elevation_file = st.file_uploader(
        "Upload elevation CSV",
        type=["csv"],
        key="elevation_upload",
    )

    if st.button(
        "Process elevation",
        disabled=elevation_file is None,
    ):

        try:

            elevation_file.seek(0)

            from mapping.map_engine import load_elevation_csv

            st.session_state.elevation = (
                load_elevation_csv(elevation_file)
            )

            st.success("Elevation data successfully loaded.")

        except Exception as error:
            display_error(error)

    if st.session_state.elevation is not None:

        st.metric(
            "Elevation samples",
            len(st.session_state.elevation),
        )

    st.divider()

    st.markdown("### Map Visualization")

    buildings = st.session_state.buildings

    st.metric(
        "Loaded building features",
        len(buildings),
    )

    if buildings:

        building_df = pd.DataFrame([
            {
                "latitude": b["lat"],
                "longitude": b["lon"],
            }
            for b in buildings
        ])

        fig = go.Figure()

        fig.add_trace(
            go.Scattergeo(
                lat=building_df["latitude"],
                lon=building_df["longitude"],
                mode="markers",
                name="Building locations",
                marker=dict(
                    size=6,
                    color="red",
                ),
            )
        )

        fig.update_layout(
            title="Building Locations",
            geo=dict(
                projection_type="mercator",
                showland=True,
            ),
            height=500,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.info("No geographic building data loaded.")

    if st.button("Clear all map data"):

        st.session_state.buildings = []
        st.session_state.elevation = None

        st.rerun()

    st.caption(
        "Building data: OpenStreetMap contributors. "
        "The current map view shows building locations, "
        "not satellite imagery or verified obstacle heights."
    )


# =====================================================
# TAB 3 - DRONE SIMULATION
# =====================================================

with tabs[2]:

    st.subheader("Drone Physics Simulation")

    st.warning(
        "Simulation-only model. It does not represent "
        "a validated full quadcopter flight controller."
    )

    col1, col2 = st.columns(2)

    with col1:

        kp = st.slider(
            "PID Kp",
            0.1,
            10.0,
            2.0,
        )

        ki = st.slider(
            "PID Ki",
            0.0,
            2.0,
            0.05,
        )

        kd = st.slider(
            "PID Kd",
            0.0,
            5.0,
            0.3,
        )

    with col2:

        duration = st.slider(
            "Simulation duration (seconds)",
            10,
            120,
            30,
        )

        dt = 0.05

    if st.button(
        "Start Simulation",
        type="primary",
    ):

        try:

            drone = Drone(
                mass=mass,
                payload=payload,
                max_thrust=max_thrust,
                battery=float(battery),
            )

            controller = PIDController(
                kp,
                ki,
                kd,
            )

            steps = int(duration / dt)

            position = np.zeros(3)
            velocity = np.zeros(3)

            wind_angle = math.radians(
                wind_direction
            )

            wind = np.array([
                wind_speed * math.cos(wind_angle),
                wind_speed * math.sin(wind_angle),
                0.0,
            ])

            records = []

            for i in range(steps):

                time = i * dt

                # Altitude controller
                correction = controller.update(
                    altitude,
                    position[2],
                    dt,
                )

                thrust_command = (
                    total_mass * 9.81
                    + correction
                )

                thrust_command = np.clip(
                    thrust_command,
                    0,
                    max_thrust,
                )

                acceleration = calculate_acceleration(
                    total_mass,
                    thrust_command,
                    0.0,
                    0.0,
                    0.0,
                    velocity,
                    wind,
                    area=drone.reference_area,
                    cd=drone.drag_coefficient,
                )

                # Numerical integration
                velocity += acceleration * dt
                position += velocity * dt

                # Ground constraint
                if position[2] < 0:

                    position[2] = 0

                    if velocity[2] < 0:
                        velocity[2] = 0

                power = (
                    thrust_command
                    * max(np.linalg.norm(velocity), 0.5)
                )

                drone.consume_energy(
                    power,
                    dt,
                )

                records.append({
                    "time": time,
                    "x": position[0],
                    "y": position[1],
                    "altitude": position[2],
                    "speed": np.linalg.norm(velocity),
                    "battery": drone.battery,
                    "thrust": thrust_command,
                })

            df = pd.DataFrame(records)

            st.session_state.flight_data = df

            st.success("Simulation completed.")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Final altitude",
                f"{df['altitude'].iloc[-1]:.2f} m",
            )

            c2.metric(
                "Final speed",
                f"{df['speed'].iloc[-1]:.2f} m/s",
            )

            c3.metric(
                "Remaining battery",
                f"{df['battery'].iloc[-1]:.2f}%",
            )

            st.markdown("### Altitude Response")

            st.plotly_chart(
                flight_chart(df),
                use_container_width=True,
            )

            st.markdown("### 3D Flight Trajectory")

            st.plotly_chart(
                make_trajectory_figure(df),
                use_container_width=True,
            )

            st.markdown("### Flight Telemetry")

            st.dataframe(
                df.tail(10),
                use_container_width=True,
            )

        except Exception as error:
            display_error(error)


# =====================================================
# TAB 4 - AI TRAINING
# =====================================================

with tabs[3]:

    st.subheader("AI Navigation Training Laboratory")

    st.info(
        "This version uses tabular Q-learning for "
        "grid-based navigation, not continuous drone control."
    )

    episodes = st.slider(
        "Training episodes",
        min_value=50,
        max_value=1000,
        value=300,
        step=50,
    )

    obstacle_probability = st.slider(
        "Obstacle probability",
        min_value=0.0,
        max_value=0.4,
        value=0.12,
    )

    if st.button(
        "Train Navigation Model",
        type="primary",
    ):

        try:

            with st.spinner("Training AI model..."):

                q_table, rewards = train_agent(
                    episodes=episodes,
                    obstacle_probability=obstacle_probability,
                )

                st.session_state.q_table = q_table

                st.session_state.training_rewards = rewards

            st.success("Training completed.")

        except Exception as error:
            display_error(error)

    if st.session_state.training_rewards is not None:

        st.markdown("### Training Performance")

        rewards = st.session_state.training_rewards

        st.plotly_chart(
            reward_chart(rewards),
            use_container_width=True,
        )

        st.metric(
            "Completed training episodes",
            len(rewards),
        )

        st.metric(
            "Average reward (last 50 episodes)",
            f"{np.mean(rewards[-50:]):.2f}",
        )

    if st.session_state.q_table is not None:

        st.markdown("### Trained Model")

        q_table = st.session_state.q_table

        st.write(
            "Q-table dimensions:",
            q_table.shape,
        )

        st.download_button(
            label="Download trained Q-table",
            data=create_download_buffer(q_table),
            file_name="aeromind_q_table.npy",
            mime="application/octet-stream",
        )

    else:

        st.info("No trained model available yet.")


# =====================================================
# TAB 5 - ANALYTICS
# =====================================================

with tabs[4]:

    st.subheader("Flight Analytics")

    df = st.session_state.flight_data

    if df is not None and not df.empty:

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Simulation duration",
            f"{df['time'].iloc[-1]:.2f} s",
        )

        c2.metric(
            "Maximum altitude",
            f"{df['altitude'].max():.2f} m",
        )

        c3.metric(
            "Maximum speed",
            f"{df['speed'].max():.2f} m/s",
        )

        st.markdown("### Altitude")

        st.line_chart(
            df.set_index("time")[["altitude"]],
        )

        st.markdown("### Velocity")

        st.line_chart(
            df.set_index("time")[["speed"]],
        )

        st.markdown("### Battery")

        st.line_chart(
            df.set_index("time")[["battery"]],
        )

        st.markdown("### Thrust")

        st.line_chart(
            df.set_index("time")[["thrust"]],
        )

        st.download_button(
            label="Download Flight Log CSV",
            data=df.to_csv(index=False),
            file_name="aeromind_flight_log.csv",
            mime="text/csv",
        )

    else:

        st.info(
            "Run a simulation to generate flight analytics."
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center">
        <b>AeroMind</b><br>
        Autonomous UAV Simulation Platform<br>
        <small>
        Experimental software | Simulation only |
        Not certified for real-world flight
        </small>
    </div>
    """,
    unsafe_allow_html=True,
)
