
import plotly.express as px
import plotly.graph_objects as go

def flight_chart(df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["altitude"],
        name="Altitude",
        mode="lines"
    ))

    fig.update_layout(
        title="Drone Altitude vs Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Altitude (m)",
        template="plotly_dark"
    )

    return fig


def trajectory_chart(df):
    fig = px.line_3d(
        df,
        x="longitude",
        y="latitude",
        z="altitude",
        title="3D Flight Trajectory"
    )

    return fig


def reward_chart(rewards):
    fig = px.line(
        x=list(range(len(rewards))),
        y=rewards,
        labels={
            "x": "Training Episode",
            "y": "Total Reward"
        },
        title="AI Training Performance"
    )

    return fig
