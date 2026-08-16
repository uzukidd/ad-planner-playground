"""Pluggable local-planning interfaces and a Frenet implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass
class LocalPlan:
    """A time-parameterized local trajectory and its Frenet metadata."""

    points: np.ndarray
    times: np.ndarray
    speeds: np.ndarray
    headings: np.ndarray
    longitudinal: float
    lateral: float
    target_shift: float
    collision_fallback: bool


class LocalPlanner(Protocol):
    """Interface required by the simulation loop and route renderer."""

    def plan(self, ego_vehicle: Any, obstacles: Sequence[Any]) -> LocalPlan: ...


@dataclass
class FrenetLocalPlanner:
    """Frenet trajectory planner using static, current-state obstacle checks."""

    horizon: float = 120.0
    obstacle_horizon: float = 70.0
    safety_gap: float = 20.0
    obstacle_margin: float = 0.35
    lateral_step_fraction: float = 1 / 5
    trajectory_duration: float = 4.0
    trajectory_dt: float = 0.2
    maneuver_duration: float = 2.0
    target_speed: float = 25.0

    @staticmethod
    def _quintic_coefficients(start: float, end: float, duration: float) -> np.ndarray:
        """Fit d(t) with zero lateral velocity and acceleration at both ends."""
        matrix = np.array(
            [
                [duration**3, duration**4, duration**5],
                [3 * duration**2, 4 * duration**3, 5 * duration**4],
                [6 * duration, 12 * duration**2, 20 * duration**3],
            ]
        )
        tail = np.linalg.solve(matrix, np.array([end - start, 0.0, 0.0]))
        return np.array([start, 0.0, 0.0, *tail])

    @staticmethod
    def _quartic_coefficients(
        start: float, start_speed: float, target_speed: float, duration: float
    ) -> np.ndarray:
        """Fit s(t) with zero longitudinal acceleration at both ends."""
        matrix = np.array(
            [
                [3 * duration**2, 4 * duration**3],
                [6 * duration, 12 * duration**2],
            ]
        )
        tail = np.linalg.solve(matrix, np.array([target_speed - start_speed, 0.0]))
        return np.array([start, start_speed, 0.0, *tail])

    @staticmethod
    def _evaluate(coefficients: np.ndarray, times: np.ndarray) -> np.ndarray:
        return sum(coefficient * times**power for power, coefficient in enumerate(coefficients))

    def _detour_lateral(
        self,
        times: np.ndarray,
        start_lateral: float,
        target_lateral: float,
        entry_time: float,
        apex_time: float,
        exit_time: float,
    ) -> np.ndarray:
        """Join two quintic lane-change segments to form a smooth detour."""
        lateral = np.full_like(times, start_lateral)
        if apex_time <= entry_time or exit_time <= apex_time:
            return lateral

        entering = (times >= entry_time) & (times <= apex_time)
        leaving = (times > apex_time) & (times <= exit_time)
        entry_coefficients = self._quintic_coefficients(
            start_lateral, target_lateral, apex_time - entry_time
        )
        exit_coefficients = self._quintic_coefficients(
            target_lateral, 0.0, exit_time - apex_time
        )
        lateral[entering] = self._evaluate(
            entry_coefficients, times[entering] - entry_time
        )
        lateral[leaving] = self._evaluate(
            exit_coefficients, times[leaving] - apex_time
        )
        lateral[times > exit_time] = 0.0
        return lateral

    def plan(self, ego_vehicle: Any, obstacles: Sequence[Any]) -> LocalPlan:
        current_lane = ego_vehicle.lane
        longitudinal, lateral = current_lane.local_coordinates(ego_vehicle.position)
        start_s = max(0.0, longitudinal)
        end_s = min(current_lane.length, start_s + self.horizon)
        times = np.arange(
            0.0,
            self.trajectory_duration + self.trajectory_dt / 2,
            self.trajectory_dt,
        )
        target_speed = self.target_speed
        if current_lane.speed_limit is not None:
            target_speed = min(target_speed, current_lane.speed_limit)
        longitudinal_coefficients = self._quartic_coefficients(
            start_s, ego_vehicle.speed, target_speed, self.trajectory_duration
        )
        longitudinal_samples = np.clip(
            self._evaluate(longitudinal_coefficients, times), start_s, end_s
        )

        lane_width = current_lane.width_at(longitudinal)
        nearby_obstacles = []
        for obstacle in obstacles:
            obstacle_s, obstacle_d = current_lane.local_coordinates(obstacle.position)
            distance = obstacle_s - longitudinal
            if (
                0.0 < distance < self.obstacle_horizon
                and abs(obstacle_d) < 1.5 * lane_width
            ):
                nearby_obstacles.append(
                    (obstacle_s, obstacle_d, obstacle.LENGTH, obstacle.WIDTH)
                )

        clearance = ego_vehicle.WIDTH / 2 + self.obstacle_margin
        current_lane_obstacles = [
            obstacle
            for obstacle in nearby_obstacles
            if abs(obstacle[1]) <= clearance + obstacle[3] / 2
        ]
        current_lane_obstacles.sort(key=lambda obstacle: obstacle[0])
        target_shift = 0.0
        collision_fallback = False
        lateral_samples = self._detour_lateral(
            times,
            lateral,
            0.0,
            0.0,
            self.trajectory_duration / 2,
            self.trajectory_duration,
        )
        if current_lane_obstacles:
            primary_obstacle_s = current_lane_obstacles[0][0]
            detour_obstacles = [
                obstacle
                for obstacle in nearby_obstacles
                if abs(obstacle[0] - primary_obstacle_s) <= self.safety_gap
            ]
            candidate_step = lane_width * self.lateral_step_fraction
            candidates = np.arange(
                -lane_width, lane_width + candidate_step / 2, candidate_step
            )
            valid_lanes = [ego_vehicle.lane_index]
            valid_lanes.extend(
                lane_index
                for lane_index in ego_vehicle.road.network.side_lanes(
                    ego_vehicle.lane_index
                )
                if ego_vehicle.road.network.get_lane(lane_index).is_reachable_from(
                    ego_vehicle.position
                )
            )

            obstacle_index = int(
                np.argmin(np.abs(longitudinal_samples - primary_obstacle_s))
            )
            apex_time = times[obstacle_index]
            half_maneuver = min(
                self.maneuver_duration / 2,
                apex_time,
                self.trajectory_duration - apex_time,
            )
            entry_time = apex_time - half_maneuver
            exit_time = apex_time + half_maneuver

            def lane_feasible(lateral_samples: np.ndarray) -> bool:
                for sample_s, sample_d in zip(longitudinal_samples, lateral_samples):
                    candidate_position = current_lane.position(sample_s, sample_d)
                    candidate_lane_index = ego_vehicle.road.network.get_closest_lane_index(
                        candidate_position, ego_vehicle.heading
                    )
                    if candidate_lane_index not in valid_lanes:
                        return False
                    if (
                        candidate_lane_index == ego_vehicle.lane_index
                        and abs(sample_d) > lane_width / 2
                    ):
                        return False
                return True

            def collision_free(lateral_samples: np.ndarray) -> bool:
                for obstacle_s, obstacle_d, obstacle_length, obstacle_width in detour_obstacles:
                    longitudinal_clearance = (
                        ego_vehicle.LENGTH / 2 + obstacle_length / 2 + self.obstacle_margin
                    )
                    lateral_clearance = (
                        ego_vehicle.WIDTH / 2 + obstacle_width / 2 + self.obstacle_margin
                    )
                    collision = (
                        np.abs(longitudinal_samples - obstacle_s)
                        < longitudinal_clearance
                    ) & (np.abs(lateral_samples - obstacle_d) < lateral_clearance)
                    if np.any(collision):
                        return False
                return True

            feasible_candidates = []
            for candidate in candidates:
                candidate_lateral = self._detour_lateral(
                    times,
                    lateral,
                    float(candidate),
                    entry_time,
                    apex_time,
                    exit_time,
                )
                if lane_feasible(candidate_lateral) and collision_free(candidate_lateral):
                    jerk = np.gradient(
                        np.gradient(
                            np.gradient(candidate_lateral, self.trajectory_dt),
                            self.trajectory_dt,
                        ),
                        self.trajectory_dt,
                    )
                    smoothness_cost = float(np.sum(jerk**2) * self.trajectory_dt)
                    feasible_candidates.append(
                        (abs(candidate) + 0.05 * smoothness_cost, candidate, candidate_lateral)
                    )
            if feasible_candidates:
                _, target_shift, lateral_samples = min(
                    feasible_candidates, key=lambda item: item[0]
                )
            else:
                collision_fallback = True
        points = np.array(
            [
                current_lane.position(float(sample_s), float(sample_d))
                for sample_s, sample_d in zip(longitudinal_samples, lateral_samples)
            ]
        )
        velocities = np.gradient(points, self.trajectory_dt, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        headings = np.arctan2(velocities[:, 1], velocities[:, 0])
        return LocalPlan(
            points,
            times,
            speeds,
            headings,
            longitudinal,
            lateral,
            target_shift,
            collision_fallback,
        )
