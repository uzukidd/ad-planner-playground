"""Pluggable behavior policies for HighwayEnv background traffic."""

from __future__ import annotations

from typing import Any, Protocol, Sequence

from highway_env.vehicle.behavior import IDMVehicle


class TrafficBehavior(Protocol):
    """Interface for configuring and advancing NPC driving behavior."""

    def reset(self, vehicles: Sequence[Any], rng: Any) -> None: ...

    def update(self, vehicles: Sequence[Any], rng: Any, dt: float) -> None: ...


class DefaultTrafficBehavior:
    """Leave HighwayEnv's built-in IDM/MOBIL traffic behavior unchanged."""

    def reset(self, vehicles: Sequence[Any], rng: Any) -> None:
        return None

    def update(self, vehicles: Sequence[Any], rng: Any, dt: float) -> None:
        return None


class RandomTrafficBehavior:
    """Give each IDM NPC a randomized style and occasional random intentions."""

    def reset(self, vehicles: Sequence[Any], rng: Any) -> None:
        for vehicle in vehicles:
            if not isinstance(vehicle, IDMVehicle):
                continue

            vehicle.DELTA = rng.uniform(3.5, 4.5)
            vehicle.TIME_WANTED = rng.uniform(0.8, 2.2)
            vehicle.DISTANCE_WANTED = rng.uniform(5.0, 10.0)
            vehicle.POLITENESS = rng.uniform(-0.2, 0.8)
            vehicle.LANE_CHANGE_MIN_ACC_GAIN = rng.uniform(0.05, 0.8)
            vehicle.LANE_CHANGE_MAX_BRAKING_IMPOSED = rng.uniform(1.5, 3.5)
            vehicle.enable_lane_change = rng.random() > 0.15
            vehicle.timer = rng.uniform(0.0, vehicle.LANE_CHANGE_DELAY)

            lane = vehicle.lane
            if lane and lane.speed_limit:
                vehicle.target_speed = rng.uniform(
                    max(15.0, lane.speed_limit - 8.0), lane.speed_limit + 3.0
                )

    def update(self, vehicles: Sequence[Any], rng: Any, dt: float) -> None:
        time_scale = dt / 0.2
        speed_change_probability = 1 - (1 - 0.15) ** time_scale
        lane_change_probability = 1 - (1 - 0.04) ** time_scale

        for vehicle in vehicles:
            if not isinstance(vehicle, IDMVehicle):
                continue

            lane = vehicle.lane
            if lane and lane.speed_limit and rng.random() < speed_change_probability:
                vehicle.target_speed = rng.uniform(12.0, lane.speed_limit + 8.0)

            if (
                rng.random() < lane_change_probability
                and vehicle.lane_index == vehicle.target_lane_index
            ):
                candidates = [
                    lane_index
                    for lane_index in vehicle.road.network.side_lanes(vehicle.lane_index)
                    if vehicle.road.network.get_lane(lane_index).is_reachable_from(
                        vehicle.position
                    )
                ]
                if candidates:
                    vehicle.target_lane_index = candidates[
                        rng.integers(0, len(candidates))
                    ]


def create_traffic_behavior(name: str) -> TrafficBehavior:
    """Create a traffic behavior selected by the demo command-line option."""
    if name == "default":
        return DefaultTrafficBehavior()
    if name == "random":
        return RandomTrafficBehavior()
    raise ValueError(f"Unknown traffic behavior: {name}")
