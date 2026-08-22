"""Pluggable controllers for tracking local trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from planning.local_planning.local_planner import LocalPlan


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


class TrajectoryController(Protocol):
    """Interface for controllers that convert a LocalPlan into an action."""

    def reset(self) -> None: ...

    def action(self, vehicle: Any, plan: LocalPlan, dt: float) -> np.ndarray: ...


@dataclass
class PIDTrajectoryController:
    """PID trajectory tracker for HighwayEnv's normalized ContinuousAction."""

    longitudinal_kp: float = 1.2
    longitudinal_ki: float = 0.05
    longitudinal_kd: float = 0.1
    lateral_kp: float = 0.4
    lateral_ki: float = 0.0
    lateral_kd: float = 0.08
    heading_weight: float = 2.0
    lookahead_time: float = 0.8
    acceleration_limit: float = 5.0
    steering_limit: float = np.pi / 4
    _speed_integral: float = 0.0
    _speed_error: float = 0.0
    _lateral_integral: float = 0.0
    _lateral_error: float = 0.0

    def reset(self) -> None:
        self._speed_integral = 0.0
        self._speed_error = 0.0
        self._lateral_integral = 0.0
        self._lateral_error = 0.0

    def action(self, vehicle: Any, plan: LocalPlan, dt: float) -> np.ndarray:
        lookahead_index = min(
            int(np.searchsorted(plan.times, self.lookahead_time)), len(plan.times) - 1
        )
        target_point = plan.points[lookahead_index]
        target_speed = plan.speeds[lookahead_index]
        target_heading = plan.headings[lookahead_index]

        if plan.collision_fallback:
            target_speed = 0.0

        speed_error = target_speed - vehicle.speed
        self._speed_integral = np.clip(
            self._speed_integral + speed_error * dt, -10.0, 10.0
        )
        speed_derivative = (speed_error - self._speed_error) / dt
        self._speed_error = speed_error
        acceleration = (
            self.longitudinal_kp * speed_error
            + self.longitudinal_ki * self._speed_integral
            + self.longitudinal_kd * speed_derivative
        )

        delta = target_point - vehicle.position
        lateral_error = -np.sin(vehicle.heading) * delta[0] + np.cos(
            vehicle.heading
        ) * delta[1]
        heading_error = wrap_angle(target_heading - vehicle.heading)
        combined_lateral_error = lateral_error + self.heading_weight * heading_error
        self._lateral_integral = np.clip(
            self._lateral_integral + combined_lateral_error * dt, -5.0, 5.0
        )
        lateral_derivative = (combined_lateral_error - self._lateral_error) / dt
        self._lateral_error = combined_lateral_error
        steering = (
            self.lateral_kp * combined_lateral_error
            + self.lateral_ki * self._lateral_integral
            + self.lateral_kd * lateral_derivative
        )

        acceleration = np.clip(
            acceleration, -self.acceleration_limit, self.acceleration_limit
        )
        steering = np.clip(steering, -self.steering_limit, self.steering_limit)
        return np.array(
            [acceleration / self.acceleration_limit, steering / self.steering_limit],
            dtype=np.float32,
        )


@dataclass
class PurePursuitTrajectoryController:
    """PID longitudinal control with Pure Pursuit lateral trajectory tracking."""

    longitudinal_kp: float = 1.2
    longitudinal_ki: float = 0.05
    longitudinal_kd: float = 0.1
    lookahead_base: float = 5.0
    lookahead_speed_gain: float = 0.5
    min_lookahead: float = 5.0
    max_lookahead: float = 20.0
    wheelbase: float = 2.5
    acceleration_limit: float = 5.0
    steering_limit: float = np.pi / 4
    _speed_integral: float = 0.0
    _speed_error: float = 0.0

    def reset(self) -> None:
        self._speed_integral = 0.0
        self._speed_error = 0.0

    def _target_index(self, vehicle: Any, plan: LocalPlan, lookahead: float) -> int:
        """Find the first point at least ``lookahead`` metres ahead on the path."""
        distances = np.linalg.norm(plan.points - vehicle.position, axis=1)
        nearest_index = int(np.argmin(distances))
        travelled = 0.0
        for index in range(nearest_index, len(plan.points) - 1):
            travelled += float(np.linalg.norm(plan.points[index + 1] - plan.points[index]))
            if travelled >= lookahead:
                return index + 1
        return len(plan.points) - 1

    def action(self, vehicle: Any, plan: LocalPlan, dt: float) -> np.ndarray:
        if plan.collision_fallback:
            self.reset()
            return np.array([-1.0, 0.0], dtype=np.float32)

        lookahead = float(
            np.clip(
                self.lookahead_base + self.lookahead_speed_gain * vehicle.speed,
                self.min_lookahead,
                self.max_lookahead,
            )
        )
        target_index = self._target_index(vehicle, plan, lookahead)
        target_point = plan.points[target_index]
        target_speed = float(plan.speeds[target_index])

        relative_target = target_point - vehicle.position
        alpha = wrap_angle(
            float(np.arctan2(relative_target[1], relative_target[0]) - vehicle.heading)
        )
        steering = np.arctan2(
            2.0 * self.wheelbase * np.sin(alpha), max(lookahead, 1e-3)
        )
        steering = float(np.clip(steering, -self.steering_limit, self.steering_limit))

        speed_error = target_speed - vehicle.speed
        self._speed_integral = np.clip(
            self._speed_integral + speed_error * dt, -10.0, 10.0
        )
        speed_derivative = (speed_error - self._speed_error) / max(dt, 1e-3)
        self._speed_error = speed_error
        acceleration = (
            self.longitudinal_kp * speed_error
            + self.longitudinal_ki * self._speed_integral
            + self.longitudinal_kd * speed_derivative
        )
        acceleration = float(
            np.clip(acceleration, -self.acceleration_limit, self.acceleration_limit)
        )
        return np.array(
            [acceleration / self.acceleration_limit, steering / self.steering_limit],
            dtype=np.float32,
        )


def _reference_at(plan: LocalPlan, time: float) -> tuple[np.ndarray, float, float]:
    index = min(int(np.searchsorted(plan.times, time)), len(plan.times) - 1)
    return plan.points[index], plan.speeds[index], plan.headings[index]
