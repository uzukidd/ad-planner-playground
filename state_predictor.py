"""Pluggable NPC state-prediction interfaces and an EKF implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np


def wrap_angle(angle: float) -> float:
    """Keep headings in the [-pi, pi] interval."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class StatePredictor(Protocol):
    """Interface required by renderers and simulation loops."""

    def reset(self, vehicles: Sequence[Any]) -> None: ...

    def update(self, vehicles: Sequence[Any], dt: float) -> None: ...

    def predict_trajectories(
        self, steps: int, dt: float
    ) -> dict[int, list[np.ndarray]]: ...


@dataclass
class NPCTrack:
    """Extended Kalman filter for a constant-turn-rate vehicle model."""

    state: np.ndarray
    covariance: np.ndarray

    @classmethod
    def from_vehicle(cls, vehicle: Any) -> "NPCTrack":
        return cls(
            state=np.array(
                [
                    vehicle.position[0],
                    vehicle.position[1],
                    vehicle.speed,
                    vehicle.heading,
                    0.0,
                ],
                dtype=float,
            ),
            covariance=np.diag([2.0, 2.0, 4.0, 0.2, 0.5]),
        )

    @staticmethod
    def transition(state: np.ndarray, dt: float) -> np.ndarray:
        x, y, speed, heading, yaw_rate = state
        return np.array(
            [
                x + speed * np.cos(heading) * dt,
                y + speed * np.sin(heading) * dt,
                speed,
                wrap_angle(heading + yaw_rate * dt),
                yaw_rate,
            ]
        )

    def predict(self, dt: float) -> None:
        _, _, speed, heading, _ = self.state
        transition_jacobian = np.array(
            [
                [1.0, 0.0, np.cos(heading) * dt, -speed * np.sin(heading) * dt, 0.0],
                [0.0, 1.0, np.sin(heading) * dt, speed * np.cos(heading) * dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        process_noise = np.diag([0.2, 0.2, 1.0, 0.02, 0.1])
        self.state = self.transition(self.state, dt)
        self.covariance = (
            transition_jacobian @ self.covariance @ transition_jacobian.T
            + process_noise
        )

    def update(self, vehicle: Any, dt: float) -> None:
        self.predict(dt)
        measurement = np.array(
            [vehicle.position[0], vehicle.position[1], vehicle.speed, vehicle.heading]
        )
        measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        measurement_noise = np.diag([0.5, 0.5, 1.0, 0.05])
        innovation = measurement - measurement_matrix @ self.state
        innovation[3] = wrap_angle(innovation[3])
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + measurement_noise
        )
        gain = np.linalg.solve(
            innovation_covariance, measurement_matrix @ self.covariance
        ).T
        self.state += gain @ innovation
        self.state[3] = wrap_angle(self.state[3])
        self.covariance = (
            np.eye(len(self.state)) - gain @ measurement_matrix
        ) @ self.covariance

    def predict_trajectory(self, steps: int, dt: float) -> list[np.ndarray]:
        predicted_state = self.state.copy()
        trajectory = [predicted_state[:2].copy()]
        for _ in range(steps):
            predicted_state = self.transition(predicted_state, dt)
            trajectory.append(predicted_state[:2].copy())
        return trajectory


@dataclass
class EKFStatePredictor:
    """Manage one EKF track per NPC vehicle."""

    tracks: dict[int, NPCTrack] = field(default_factory=dict)

    def reset(self, vehicles: Sequence[Any]) -> None:
        self.tracks = {id(vehicle): NPCTrack.from_vehicle(vehicle) for vehicle in vehicles}

    def update(self, vehicles: Sequence[Any], dt: float) -> None:
        active_ids = set()
        for vehicle in vehicles:
            vehicle_id = id(vehicle)
            active_ids.add(vehicle_id)
            if vehicle_id not in self.tracks:
                self.tracks[vehicle_id] = NPCTrack.from_vehicle(vehicle)
            else:
                self.tracks[vehicle_id].update(vehicle, dt)

        for vehicle_id in set(self.tracks) - active_ids:
            del self.tracks[vehicle_id]

    def predict_trajectories(
        self, steps: int, dt: float
    ) -> dict[int, list[np.ndarray]]:
        return {
            vehicle_id: track.predict_trajectory(steps, dt)
            for vehicle_id, track in self.tracks.items()
        }
