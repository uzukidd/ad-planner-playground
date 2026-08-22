"""Pluggable NPC state-prediction interfaces and an EKF implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np


def wrap_angle(angle: float) -> float:
    """Keep headings in the [-pi, pi] interval."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class TimedObstacleState:
    """Obstacle state expressed relative to the current planning instant."""

    vehicle_id: int
    position: np.ndarray
    speed: float
    heading: float
    length: float
    width: float
    time: float = 0.0

    @property
    def LENGTH(self) -> float:
        """HighwayEnv-compatible vehicle length for existing planners."""
        return self.length

    @property
    def WIDTH(self) -> float:
        """HighwayEnv-compatible vehicle width for existing planners."""
        return self.width

    @classmethod
    def from_vehicle(cls, vehicle: Any, time: float = 0.0) -> "TimedObstacleState":
        return cls(
            vehicle_id=id(vehicle),
            position=np.asarray(vehicle.position, dtype=float).copy(),
            speed=float(vehicle.speed),
            heading=float(vehicle.heading),
            length=float(vehicle.LENGTH),
            width=float(vehicle.WIDTH),
            time=float(time),
        )


@dataclass(frozen=True)
class TimedObstacleTrajectory:
    """Time-ordered states for one obstacle relative to one planning cycle."""

    states: tuple[TimedObstacleState, ...]
    time_array: np.ndarray = field(init=False, repr=False)
    position_array: np.ndarray = field(init=False, repr=False)
    length_array: np.ndarray = field(init=False, repr=False)
    width_array: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("An obstacle trajectory needs at least one state.")
        if any(
            current.time > following.time
            for current, following in zip(self.states, self.states[1:])
        ):
            raise ValueError("Obstacle states must be sorted by nondecreasing time.")
        object.__setattr__(self, "time_array", np.asarray([state.time for state in self.states], dtype=float))
        object.__setattr__(self, "position_array", np.asarray([state.position for state in self.states], dtype=float))
        object.__setattr__(self, "length_array", np.asarray([state.length for state in self.states], dtype=float))
        object.__setattr__(self, "width_array", np.asarray([state.width for state in self.states], dtype=float))

    def state_at(self, time: float) -> TimedObstacleState:
        """Return the last state whose timestamp is not greater than ``time``."""
        index = int(np.searchsorted(self.time_array, time, side="right") - 1)
        return self.states[max(0, index)]

    def indices_at(self, times: np.ndarray) -> np.ndarray:
        """Return the latest state index not later than each requested time."""
        return np.maximum(
            np.searchsorted(self.time_array, times, side="right") - 1,
            0,
        )

    @property
    def position(self) -> np.ndarray:
        return self.state_at(0.0).position

    @property
    def speed(self) -> float:
        return self.state_at(0.0).speed

    @property
    def heading(self) -> float:
        return self.state_at(0.0).heading

    @property
    def LENGTH(self) -> float:
        return self.state_at(0.0).length

    @property
    def WIDTH(self) -> float:
        return self.state_at(0.0).width


class ObstacleStatePredictor(Protocol):
    """Convert detected obstacles into planner-compatible timed states."""

    def reset(self, obstacles: Sequence[Any]) -> None: ...

    def update(self, obstacles: Sequence[Any], dt: float) -> None: ...

    def predict(
        self, obstacles: Sequence[Any], times: np.ndarray
    ) -> tuple[TimedObstacleTrajectory, ...]: ...


@dataclass
class CurrentStatePredictor:
    """Pass through current obstacle states without future extrapolation."""

    def reset(self, obstacles: Sequence[Any]) -> None:
        return None

    def update(self, obstacles: Sequence[Any], dt: float) -> None:
        return None

    def predict(
        self, obstacles: Sequence[Any], times: np.ndarray
    ) -> tuple[TimedObstacleTrajectory, ...]:
        return tuple(
            TimedObstacleTrajectory(
                tuple(
                    TimedObstacleState.from_vehicle(obstacle, float(time))
                    for time in times
                )
            )
            for obstacle in obstacles
        )


@dataclass
class NPCTrack:
    """Algorithm-independent ground-truth observation history for one NPC."""

    vehicle_id: int
    position: np.ndarray
    speed: float
    heading: float
    length: float
    width: float
    history: list[np.ndarray] = field(default_factory=list)
    history_limit: int = 20

    @classmethod
    def from_vehicle(cls, vehicle: Any) -> "NPCTrack":
        position = np.asarray(vehicle.position, dtype=float).copy()
        return cls(
            id(vehicle),
            position,
            float(vehicle.speed),
            float(vehicle.heading),
            float(vehicle.LENGTH),
            float(vehicle.WIDTH),
            [position],
        )

    def update(self, vehicle: Any) -> None:
        self.position = np.asarray(vehicle.position, dtype=float).copy()
        self.speed = float(vehicle.speed)
        self.heading = float(vehicle.heading)
        self.length = float(vehicle.LENGTH)
        self.width = float(vehicle.WIDTH)
        self.history.append(self.position.copy())
        if len(self.history) > self.history_limit:
            del self.history[: len(self.history) - self.history_limit]


class TrackPredictionModel(Protocol):
    """Prediction algorithm operating on one algorithm-independent NPC track."""

    def reset(self, track: NPCTrack) -> None: ...

    def update(self, track: NPCTrack, dt: float) -> None: ...

    def predict_trajectory(self, steps: int, dt: float) -> list[np.ndarray]: ...


class StatePredictor(Protocol):
    """Interface required by renderers and simulation loops."""

    def reset(self, vehicles: Sequence[Any]) -> None: ...

    def update(self, vehicles: Sequence[Any], dt: float) -> None: ...

    def predict_trajectories(
        self, steps: int, dt: float
    ) -> dict[int, list[np.ndarray]]: ...


@dataclass
class EKFTrackPredictionModel:
    """Extended Kalman filter for a constant-turn-rate vehicle model."""

    initial_covariance: tuple[float, ...] = (2.0, 2.0, 4.0, 0.2, 0.5)
    process_noise: tuple[float, ...] = (0.2, 0.2, 1.0, 0.02, 0.1)
    measurement_noise: tuple[float, ...] = (0.5, 0.5, 1.0, 0.05)
    state: np.ndarray | None = field(default=None, init=False)
    covariance: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate diagonal covariance and noise vectors from configuration."""
        if len(self.initial_covariance) != 5:
            raise ValueError("initial_covariance must contain 5 diagonal values.")
        if len(self.process_noise) != 5:
            raise ValueError("process_noise must contain 5 diagonal values.")
        if len(self.measurement_noise) != 4:
            raise ValueError("measurement_noise must contain 4 diagonal values.")
        for name, values in (
            ("initial_covariance", self.initial_covariance),
            ("process_noise", self.process_noise),
            ("measurement_noise", self.measurement_noise),
        ):
            if not np.all(np.isfinite(values)) or np.any(np.asarray(values) < 0.0):
                raise ValueError(f"{name} must contain finite nonnegative values.")

    def reset(self, track: NPCTrack) -> None:
        self.state = np.array(
            [track.position[0], track.position[1], track.speed, track.heading, 0.0],
            dtype=float,
        )
        self.covariance = np.diag(self.initial_covariance)

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
        if self.state is None or self.covariance is None:
            raise RuntimeError("Prediction model must be reset before use.")
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
        process_noise = np.diag(self.process_noise)
        self.state = self.transition(self.state, dt)
        self.covariance = (
            transition_jacobian @ self.covariance @ transition_jacobian.T
            + process_noise
        )

    def update(self, track: NPCTrack, dt: float) -> None:
        self.predict(dt)
        if self.state is None or self.covariance is None:
            raise RuntimeError("Prediction model must be reset before use.")
        measurement = np.array(
            [track.position[0], track.position[1], track.speed, track.heading]
        )
        measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
        measurement_noise = np.diag(self.measurement_noise)
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
        if self.state is None:
            raise RuntimeError("Prediction model must be reset before use.")
        predicted_state = self.state.copy()
        trajectory = [predicted_state[:2].copy()]
        for _ in range(steps):
            predicted_state = self.transition(predicted_state, dt)
            trajectory.append(predicted_state[:2].copy())
        return trajectory

    def predict_states(self, times: np.ndarray) -> np.ndarray:
        """Predict complete EKF states at the requested relative timestamps."""
        if self.state is None:
            raise RuntimeError("Prediction model must be reset before use.")
        requested_times = np.asarray(times, dtype=float)
        if np.any(requested_times < 0) or np.any(np.diff(requested_times) < 0):
            raise ValueError("Prediction times must be nonnegative and sorted.")

        predicted_state = self.state.copy()
        predicted_states = []
        previous_time = 0.0
        for time in requested_times:
            predicted_state = self.transition(
                predicted_state, float(time - previous_time)
            )
            predicted_states.append(predicted_state.copy())
            previous_time = float(time)
        return np.asarray(predicted_states)


@dataclass
class MultiTrackStatePredictor:
    """Manage tracks and delegate prediction to an injectable model."""

    model_factory: Callable[[], TrackPredictionModel]
    tracks: dict[int, NPCTrack] = field(default_factory=dict)
    models: dict[int, TrackPredictionModel] = field(default_factory=dict)

    def reset(self, vehicles: Sequence[Any]) -> None:
        self.tracks = {}
        self.models = {}
        for vehicle in vehicles:
            self._add_vehicle(vehicle)

    def _add_vehicle(self, vehicle: Any) -> None:
        track = NPCTrack.from_vehicle(vehicle)
        model = self.model_factory()
        model.reset(track)
        self.tracks[track.vehicle_id] = track
        self.models[track.vehicle_id] = model

    def update(self, vehicles: Sequence[Any], dt: float) -> None:
        active_ids = set()
        for vehicle in vehicles:
            vehicle_id = id(vehicle)
            active_ids.add(vehicle_id)
            if vehicle_id not in self.tracks:
                self._add_vehicle(vehicle)
                continue
            track = self.tracks[vehicle_id]
            track.update(vehicle)
            self.models[vehicle_id].update(track, dt)

        for vehicle_id in set(self.tracks) - active_ids:
            del self.tracks[vehicle_id]
            del self.models[vehicle_id]

    def predict_trajectories(
        self, steps: int, dt: float
    ) -> dict[int, list[np.ndarray]]:
        return {
            vehicle_id: model.predict_trajectory(steps, dt)
            for vehicle_id, model in self.models.items()
        }


class EKFStatePredictor(MultiTrackStatePredictor):
    """Backward-compatible alias for the default EKF configuration."""

    def __init__(self, **model_parameters: Any) -> None:
        super().__init__(
            model_factory=lambda: EKFTrackPredictionModel(**model_parameters)
        )


class EKFObstacleStatePredictor(MultiTrackStatePredictor):
    """Produce time-aligned obstacle trajectories from per-NPC EKF models."""

    def __init__(self, **model_parameters: Any) -> None:
        super().__init__(
            model_factory=lambda: EKFTrackPredictionModel(**model_parameters)
        )

    def predict(
        self, obstacles: Sequence[Any], times: np.ndarray
    ) -> tuple[TimedObstacleTrajectory, ...]:
        active_ids = {id(obstacle) for obstacle in obstacles}
        for obstacle in obstacles:
            if id(obstacle) not in self.tracks:
                self._add_vehicle(obstacle)
        for vehicle_id in set(self.tracks) - active_ids:
            del self.tracks[vehicle_id]
            del self.models[vehicle_id]

        trajectories = []
        for obstacle in obstacles:
            vehicle_id = id(obstacle)
            track = self.tracks[vehicle_id]
            model = self.models[vehicle_id]
            if not isinstance(model, EKFTrackPredictionModel):
                raise TypeError("EKFObstacleStatePredictor requires EKF track models.")
            predicted_states = model.predict_states(times)
            states = tuple(
                TimedObstacleState(
                    vehicle_id=vehicle_id,
                    position=state[:2].copy(),
                    speed=float(state[2]),
                    heading=float(state[3]),
                    length=track.length,
                    width=track.width,
                    time=float(time),
                )
                for time, state in zip(times, predicted_states)
            )
            trajectories.append(TimedObstacleTrajectory(states))
        return tuple(trajectories)


def create_state_predictor(
    name: str,
    config: dict[str, Any] | None = None,
) -> StatePredictor:
    """Create a registered multi-NPC prediction manager."""
    if name == "ekf":
        predictor_config = (config or {}).get("state_predictors", {}).get("ekf", {})
        return EKFObstacleStatePredictor(**predictor_config)
    raise ValueError(f"Unknown state predictor: {name}")
