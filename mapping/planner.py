
import math
from shapely.geometry import LineString

EARTH_RADIUS = 6371000

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS * math.asin(
        math.sqrt(a)
    )


def route_distance(waypoints):
    total = 0.0

    for i in range(len(waypoints) - 1):
        total += haversine(
            waypoints[i][0],
            waypoints[i][1],
            waypoints[i+1][0],
            waypoints[i+1][1]
        )

    return total


def check_building_intersections(
    waypoints,
    buildings
):
    """
    Checks a straight-line route against
    available building geometries.

    Coordinates are longitude, latitude.
    """

    route = LineString([
        (lon, lat) for lat, lon in waypoints
    ])

    collisions = []

    for index, building in enumerate(buildings):
        if building["geometry"]:
            polygon = LineString(
                building["geometry"]
            )

            if polygon.is_ring:
                from shapely.geometry import Polygon
                polygon = Polygon(polygon)

            if route.intersects(polygon):
                collisions.append(index)

    return collisions
