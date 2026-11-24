
# AeroMind

Autonomous UAV Navigation and Digital Twin Platform.

## Features

- Streamlit web dashboard
- Configurable quadcopter
- Simplified 3D translational physics
- PID altitude control
- Wind disturbance modelling
- OpenStreetMap building retrieval
- GeoJSON upload
- Elevation CSV processing
- Geographic distance calculation
- Building intersection checks
- Experimental Q-learning navigation
- Flight analytics and CSV export

## Deployment

1. Create a GitHub repository named AeroMind.
2. Upload the complete project folder.
3. Open https://share.streamlit.io/
4. Connect your GitHub account.
5. Select the AeroMind repository.
6. Select app.py as the main file.
7. Deploy.

## Local development

Install dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py

## Limitations

This is an experimental simulation platform.

It does not implement certified flight control,
complete 6-DOF dynamics, validated obstacle avoidance,
or real-world autonomous navigation.

The Q-learning model is a grid-navigation experiment,
not a trained physical drone controller.

Map data availability depends on external providers.
