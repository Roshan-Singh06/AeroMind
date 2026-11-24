
import numpy as np

GRAVITY = 9.81

def calculate_acceleration(
    mass,
    thrust,
    pitch,
    roll,
    yaw,
    velocity,
    wind,
    air_density=1.225,
    cd=0.8,
    area=0.12
):
    """
    Simplified 3D quadcopter translational model.

    Angles are in radians.
    Coordinates: x, y, z.
    """

    relative_velocity = velocity - wind
    speed = np.linalg.norm(relative_velocity)

    if speed > 1e-9:
        drag = (
            -0.5 * air_density * cd * area
            * speed * relative_velocity
        )
    else:
        drag = np.zeros(3)

    # Body thrust transformed into world coordinates.
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rotation = np.array([
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp,   cp*sr,           cp*cr]
    ])

    thrust_world = rotation @ np.array([0, 0, thrust])

    gravity = np.array([0, 0, -mass * GRAVITY])

    acceleration = (
        thrust_world + drag + gravity
    ) / mass

    return acceleration
