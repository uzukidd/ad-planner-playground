"""Pluggable local-planning interfaces and a Frenet implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from .global_planner import GlobalPlan


@dataclass
class CandidateTrajectory:
    """A sampled Frenet candidate retained for planning visualization."""

    points: np.ndarray
    target_shift: float
    collision: bool
    lane_feasible: bool
    times: np.ndarray | None = None
    longitudinal: np.ndarray | None = None
    lateral: np.ndarray | None = None
    cost: float = float("inf")
    duration: float = 0.0
    target_speed: float = 0.0


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
    candidates: tuple[CandidateTrajectory, ...] = ()


@dataclass
class _FrenetObstacle:
    """Cached obstacle samples in the current lane's Frenet frame."""

    source: Any
    times: np.ndarray
    longitudinal: np.ndarray
    lateral: np.ndarray
    lengths: np.ndarray
    widths: np.ndarray
    speeds: np.ndarray

    def indices_at(self, times: np.ndarray) -> np.ndarray:
        if hasattr(self.source, "indices_at"):
            return self.source.indices_at(times)
        return np.zeros(len(times), dtype=int)


class LocalPlanner(Protocol):
    """Interface required by the simulation loop and route renderer."""

    def plan(
        self,
        ego_vehicle: Any,
        obstacles: Sequence[Any],
        ego_state: str = "cruise",
        reference_route: GlobalPlan | None = None,
    ) -> LocalPlan: ...

    def prediction_times(self) -> np.ndarray: ...


@dataclass
class FrenetLocalPlanner:
    """Frenet planner using sampled terminal states and polynomial candidates."""

    horizon: float = 120.0
    obstacle_horizon: float = 70.0
    safety_gap: float = 20.0
    obstacle_margin: float = 0.35
    lateral_step_fraction: float = 1 / 5
    trajectory_duration: float = 4.0
    trajectory_dt: float = 0.2
    maneuver_duration: float = 2.0
    target_speed: float = 70.0
    target_speed_mps: float = field(init=False)
    duration_samples: tuple[float, ...] = (3.0, 4.0, 5.0)
    lane_boundary_tolerance: float = 1e-6
    footprint_check_substeps: int = 4
    follow_time_headway: float = 1.5
    follow_min_gap: float = 8.0
    stop_gap: float = 5.0

    def __post_init__(self) -> None:
        """Convert the configured target speed from km/h to internal m/s."""
        if not np.isfinite(self.target_speed) or self.target_speed < 0.0:
            raise ValueError("target_speed must be a finite nonnegative value in km/h.")
        self.target_speed_mps = float(self.target_speed) / 3.6

    def prediction_times(self) -> np.ndarray:
        """Timestamps shared by Ego planning and obstacle prediction."""
        maximum_duration = max(self._candidate_durations())
        return self._sample_times(maximum_duration)

    def _candidate_durations(self) -> tuple[float, ...]:
        """Return sorted positive terminal times, including the legacy value."""
        durations = {float(self.trajectory_duration)}
        durations.update(float(duration) for duration in self.duration_samples)
        return tuple(sorted(duration for duration in durations if duration > 0.0))

    @staticmethod
    def _quintic_coefficients(
        start: float,
        end: float,
        duration: float,
        start_velocity: float = 0.0,
        start_acceleration: float = 0.0,
        end_velocity: float = 0.0,
        end_acceleration: float = 0.0,
    ) -> np.ndarray:
        """Fit a quintic from the current lateral state to a terminal state."""
        if duration <= 0.0:
            raise ValueError("Quintic duration must be positive.")
        matrix = np.array(
            [
                [duration**3, duration**4, duration**5],
                [3 * duration**2, 4 * duration**3, 5 * duration**4],
                [6 * duration, 12 * duration**2, 20 * duration**3],
            ]
        )
        tail = np.linalg.solve(
            matrix,
            np.array(
                [
                    end - start - start_velocity * duration
                    - 0.5 * start_acceleration * duration**2,
                    end_velocity - start_velocity - start_acceleration * duration,
                    end_acceleration - start_acceleration,
                ]
            ),
        )
        return np.array(
            [start, start_velocity, 0.5 * start_acceleration, *tail]
        )

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
    def _longitudinal_quintic_coefficients(
        start: float,
        start_speed: float,
        start_acceleration: float,
        end: float,
        end_speed: float,
        end_acceleration: float,
        duration: float,
    ) -> np.ndarray:
        """Fit s(t) to a terminal position, speed, and acceleration."""
        matrix = np.array(
            [
                [duration**3, duration**4, duration**5],
                [3 * duration**2, 4 * duration**3, 5 * duration**4],
                [6 * duration, 12 * duration**2, 20 * duration**3],
            ]
        )
        tail = np.linalg.solve(
            matrix,
            np.array(
                [
                    end - start - start_speed * duration
                    - 0.5 * start_acceleration * duration**2,
                    end_speed - start_speed - start_acceleration * duration,
                    end_acceleration - start_acceleration,
                ]
            ),
        )
        return np.array(
            [start, start_speed, 0.5 * start_acceleration, *tail]
        )

    @staticmethod
    def _evaluate(coefficients: np.ndarray, times: np.ndarray) -> np.ndarray:
        return sum(coefficient * times**power for power, coefficient in enumerate(coefficients))

    @staticmethod
    def _local_coordinates_array(lane: Any, positions: np.ndarray) -> np.ndarray:
        """Vectorized lane coordinates for HighwayEnv's common lane types."""
        positions = np.asarray(positions, dtype=float)
        if all(
            hasattr(lane, attribute)
            for attribute in ("start", "direction", "direction_lateral")
        ):
            delta = positions - np.asarray(lane.start, dtype=float)
            return np.column_stack(
                (
                    delta @ np.asarray(lane.direction, dtype=float),
                    delta @ np.asarray(lane.direction_lateral, dtype=float),
                )
            )
        if all(
            hasattr(lane, attribute)
            for attribute in ("center", "direction", "radius", "start_phase")
        ):
            delta = positions - np.asarray(lane.center, dtype=float)
            phi = np.arctan2(delta[:, 1], delta[:, 0])
            phase_delta = (phi - lane.start_phase + np.pi) % (2 * np.pi) - np.pi
            phi = lane.start_phase + phase_delta
            radius = np.linalg.norm(delta, axis=1)
            longitudinal = lane.direction * phase_delta * lane.radius
            lateral = lane.direction * (lane.radius - radius)
            return np.column_stack((longitudinal, lateral))
        return np.asarray([lane.local_coordinates(position) for position in positions])

    @staticmethod
    def _positions_array(
        lane: Any, longitudinal: np.ndarray, lateral: np.ndarray
    ) -> np.ndarray:
        """Vectorized lane positions for the common HighwayEnv lane types."""
        longitudinal = np.asarray(longitudinal, dtype=float)
        lateral = np.asarray(lateral, dtype=float)
        if all(
            hasattr(lane, attribute)
            for attribute in ("start", "direction", "direction_lateral")
        ):
            return (
                np.asarray(lane.start, dtype=float)
                + longitudinal[:, None] * np.asarray(lane.direction, dtype=float)
                + lateral[:, None]
                * np.asarray(lane.direction_lateral, dtype=float)
            )
        if all(
            hasattr(lane, attribute)
            for attribute in ("center", "direction", "radius", "start_phase")
        ):
            phi = lane.direction * longitudinal / lane.radius + lane.start_phase
            radius = lane.radius - lateral * lane.direction
            return np.column_stack(
                (
                    lane.center[0] + radius * np.cos(phi),
                    lane.center[1] + radius * np.sin(phi),
                )
            )
        return np.asarray(
            [
                lane.position(float(sample_s), float(sample_d))
                for sample_s, sample_d in zip(longitudinal, lateral)
            ]
        )

    def _sample_times(self, duration: float) -> np.ndarray:
        if duration <= 0.0 or self.trajectory_dt <= 0.0:
            raise ValueError("Trajectory duration and dt must be positive.")
        step_count = int(np.floor(duration / self.trajectory_dt + 1e-9))
        times = np.arange(step_count + 1, dtype=float) * self.trajectory_dt
        if times[-1] < duration - 1e-9:
            times = np.append(times, duration)
        else:
            times[-1] = duration
        return times

    @staticmethod
    def _derivative(
        coefficients: np.ndarray, times: np.ndarray, order: int
    ) -> np.ndarray:
        derivative = coefficients.copy()
        for _ in range(order):
            derivative = np.array(
                [power * value for power, value in enumerate(derivative)][1:]
            )
        if len(derivative) == 0:
            return np.zeros_like(times)
        return FrenetLocalPlanner._evaluate(derivative, times)

    def _candidate_cost(
        self,
        lateral_coefficients: np.ndarray,
        longitudinal_coefficients: np.ndarray,
        times: np.ndarray,
        target_lateral: float,
        target_speed: float,
    ) -> float:
        """Score comfort, duration, centerline offset, and terminal speed."""
        lateral_jerk = self._derivative(lateral_coefficients, times, 3)
        longitudinal_jerk = self._derivative(longitudinal_coefficients, times, 3)
        return float(
            0.05 * np.sum(lateral_jerk**2) * self.trajectory_dt
            + 0.005 * np.sum(longitudinal_jerk**2) * self.trajectory_dt
            + 0.1 * times[-1]
            + 0.5 * target_lateral**2
            + 0.02 * (target_speed - self.target_speed_mps) ** 2
        )

    def _frenet_headings(
        self,
        lane: Any,
        longitudinal: np.ndarray,
        lateral: np.ndarray,
        times: np.ndarray,
        fallback_heading: float,
    ) -> np.ndarray:
        """Compute vehicle headings without being fooled by clipped samples."""
        longitudinal = np.asarray(longitudinal, dtype=float)
        lateral = np.asarray(lateral, dtype=float)
        times = np.asarray(times, dtype=float)
        longitudinal_speed = np.gradient(longitudinal, times)
        lateral_speed = np.gradient(lateral, times)
        relative_headings = np.arctan2(lateral_speed, longitudinal_speed)
        # A clipped longitudinal sample can have s_dot=0 while d_dot is
        # still nonzero.  That is a planner boundary artifact, not a vehicle
        # pointing sideways, so continue the last valid forward heading.
        valid = np.abs(longitudinal_speed) > 1e-6
        for index in range(len(relative_headings)):
            if not valid[index]:
                relative_headings[index] = (
                    relative_headings[index - 1] if index > 0 else 0.0
                )
        lane_headings = np.asarray(
            [
                lane.heading_at(
                    float(np.clip(sample, 0.0, lane.length))
                )
                for sample in longitudinal
            ],
            dtype=float,
        )
        headings = lane_headings + relative_headings
        if len(headings) and not np.isfinite(headings[0]):
            headings[0] = fallback_heading
        return headings

    def plan(
        self,
        ego_vehicle: Any,
        obstacles: Sequence[Any],
        ego_state: str = "cruise",
        reference_route: GlobalPlan | None = None,
    ) -> LocalPlan:
        if ego_state not in {"cruise", "follow", "stop"}:
            raise ValueError(f"Unsupported ego state: {ego_state}")
        current_lane = (
            reference_route.lane if reference_route is not None else ego_vehicle.lane
        )
        reference_lane_index = (
            reference_route.lane_index
            if reference_route is not None
            else ego_vehicle.lane_index
        )
        longitudinal, lateral = current_lane.local_coordinates(ego_vehicle.position)
        start_s = max(0.0, longitudinal)
        end_s = min(current_lane.length, start_s + self.horizon)
        target_speed = self.target_speed_mps
        if current_lane.speed_limit is not None:
            target_speed = min(target_speed, current_lane.speed_limit)

        lane_width = current_lane.width_at(longitudinal)
        nearby_obstacles = []
        for obstacle in obstacles:
            obstacle_states = getattr(obstacle, "states", (obstacle,))
            obstacle_times = getattr(
                obstacle,
                "time_array",
                np.zeros(len(obstacle_states), dtype=float),
            )
            obstacle_positions = getattr(
                obstacle,
                "position_array",
                np.asarray([state.position for state in obstacle_states], dtype=float),
            )
            obstacle_lengths = getattr(
                obstacle,
                "length_array",
                np.asarray([state.LENGTH for state in obstacle_states], dtype=float),
            )
            obstacle_widths = getattr(
                obstacle,
                "width_array",
                np.asarray([state.WIDTH for state in obstacle_states], dtype=float),
            )
            frenet_coordinates = self._local_coordinates_array(
                current_lane, obstacle_positions
            )
            cached_obstacle = _FrenetObstacle(
                obstacle,
                obstacle_times,
                frenet_coordinates[:, 0],
                frenet_coordinates[:, 1],
                obstacle_lengths,
                obstacle_widths,
                np.asarray([state.speed for state in obstacle_states], dtype=float),
            )
            in_planning_area = False
            for obstacle_s, obstacle_d in frenet_coordinates:
                distance = obstacle_s - longitudinal
                if (
                    -self.obstacle_horizon < distance < self.obstacle_horizon
                    and abs(obstacle_d) < 2.0 * lane_width
                ):
                    in_planning_area = True
                    break
            if in_planning_area:
                nearby_obstacles.append(cached_obstacle)

        lead_obstacle = None
        lead_distance = float("inf")
        for obstacle in nearby_obstacles:
            obstacle_s = float(obstacle.longitudinal[0])
            obstacle_d = float(obstacle.lateral[0])
            if (
                obstacle_s > start_s
                and abs(obstacle_d - lateral) < lane_width / 2.0
                and obstacle_s - start_s < lead_distance
            ):
                lead_obstacle = obstacle
                lead_distance = obstacle_s - start_s

        # Cruise, follow, and stop all keep the current lane.  Lateral
        # detours are intentionally not a cruise behavior.
        lateral_targets = np.array([0.0])

        def collision_free(
            candidate_longitudinal: np.ndarray,
            candidate_lateral: np.ndarray,
            candidate_times: np.ndarray,
            obstacles_to_check: Sequence[_FrenetObstacle],
        ) -> bool:
            """Check time-indexed obstacle geometry using cached Frenet arrays."""
            for obstacle in obstacles_to_check:
                indices = obstacle.indices_at(candidate_times)
                obstacle_s = obstacle.longitudinal[indices]
                obstacle_d = obstacle.lateral[indices]
                longitudinal_clearance = (
                    ego_vehicle.LENGTH / 2
                    + obstacle.lengths[indices] / 2
                    + self.obstacle_margin
                )
                lateral_clearance = (
                    ego_vehicle.WIDTH / 2
                    + obstacle.widths[indices] / 2
                    + self.obstacle_margin
                )
                collision = (
                    np.abs(candidate_longitudinal - obstacle_s)
                    < longitudinal_clearance
                ) & (
                    np.abs(candidate_lateral - obstacle_d) < lateral_clearance
                )
                if np.any(collision):
                    return False
            return True

        valid_lanes = [reference_lane_index]
        valid_lanes.extend(
            lane_index
            for lane_index in ego_vehicle.road.network.all_side_lanes(
                reference_lane_index
            )
            if lane_index != reference_lane_index
        )
        legal_lanes = [
            ego_vehicle.road.network.get_lane(lane_index)
            for lane_index in valid_lanes
        ]
        fast_lane_geometry = all(
            all(
                hasattr(lane, attribute)
                for attribute in ("start", "direction", "direction_lateral")
            )
            for lane in legal_lanes
        )
        if fast_lane_geometry:
            lane_bounds = []
            for lane in legal_lanes:
                lane_longitudinal = np.clip(longitudinal, 0.0, lane.length)
                lane_center = lane.position(float(lane_longitudinal), 0.0)
                # Convert the other lane's center into the current reference
                # lane frame.  Converting the reference point into the other
                # lane frame reverses the lateral sign on edge lanes.
                _, lane_lateral = current_lane.local_coordinates(lane_center)
                half_width = lane.width_at(lane_longitudinal) / 2.0
                lane_bounds.append(
                    (lane_lateral - half_width, lane_lateral + half_width)
                )
            legal_lateral_min = min(bound[0] for bound in lane_bounds)
            legal_lateral_max = max(bound[1] for bound in lane_bounds)

        def lane_feasible(
            candidate_points: np.ndarray,
            candidate_times: np.ndarray,
            candidate_longitudinal: np.ndarray,
            candidate_lateral: np.ndarray,
        ) -> bool:
            """Check the complete ego rectangle against the legal lane union.

            Checking only the trajectory center can accept a path whose body
            crosses the road boundary.  Each sampled vehicle footprint corner
            must be inside at least one of the legal lanes.  Adjacent lanes
            form a continuous union, so corners may belong to different lanes.
            """
            headings = self._frenet_headings(
                current_lane,
                candidate_longitudinal,
                candidate_lateral,
                candidate_times,
                ego_vehicle.heading,
            )

            half_length = float(ego_vehicle.LENGTH) / 2.0
            half_width = float(ego_vehicle.WIDTH) / 2.0
            tolerance = self.lane_boundary_tolerance
            forward = half_length * np.column_stack(
                (np.cos(headings), np.sin(headings))
            )
            lateral_axis = half_width * np.column_stack(
                (-np.sin(headings), np.cos(headings))
            )
            corners = np.stack(
                (
                    candidate_points + forward + lateral_axis,
                    candidate_points + forward - lateral_axis,
                    candidate_points - forward + lateral_axis,
                    candidate_points - forward - lateral_axis,
                ),
                axis=1,
            )
            flattened_corners = corners.reshape(-1, 2)

            if fast_lane_geometry:
                coordinates = self._local_coordinates_array(
                    current_lane, flattened_corners
                ).reshape(len(candidate_points), 4, 2)
                longitudinal_ok = (
                    (coordinates[:, :, 0] >= -tolerance)
                    & (coordinates[:, :, 0] <= current_lane.length + tolerance)
                )
                lateral_ok = (
                    (coordinates[:, :, 1] >= legal_lateral_min - tolerance)
                    & (coordinates[:, :, 1] <= legal_lateral_max + tolerance)
                )
                return bool(np.all(longitudinal_ok & lateral_ok))

            for corner in flattened_corners:
                corner_is_legal = False
                for lane in legal_lanes:
                    corner_s, corner_d = lane.local_coordinates(corner)
                    if (
                        -tolerance <= corner_s <= lane.length + tolerance
                        and abs(corner_d)
                        <= lane.width_at(corner_s) / 2.0 + tolerance
                    ):
                        corner_is_legal = True
                        break
                if not corner_is_legal:
                    return False
            return True

        # Generate every (T, d_T) pair even when the centerline is currently clear.
        candidate_trajectories: list[CandidateTrajectory] = []
        feasible_candidates: list[tuple[float, CandidateTrajectory]] = []
        for duration in self._candidate_durations():
            candidate_times = self._sample_times(duration)
            longitudinal_coefficients = self._quartic_coefficients(
                start_s, ego_vehicle.speed, target_speed, duration
            )
            candidate_target_speed = target_speed
            if ego_state == "stop":
                if lead_obstacle is not None:
                    lead_indices = lead_obstacle.indices_at(candidate_times)
                    lead_s_at_end = float(lead_obstacle.longitudinal[lead_indices[-1]])
                    lead_length_at_end = float(lead_obstacle.lengths[lead_indices[-1]])
                    desired_gap = max(
                        self.stop_gap,
                        ego_vehicle.LENGTH / 2
                        + lead_length_at_end / 2
                        + self.obstacle_margin,
                    )
                    target_longitudinal = np.clip(
                        lead_s_at_end - desired_gap,
                        start_s,
                        end_s,
                    )
                else:
                    braking_acceleration = max(1.0, 0.5 * self.obstacle_margin + 3.0)
                    stopping_distance = ego_vehicle.speed**2 / (2.0 * braking_acceleration)
                    target_longitudinal = np.clip(
                        start_s + stopping_distance,
                        start_s,
                        end_s,
                    )
                candidate_target_speed = 0.0
                longitudinal_coefficients = self._longitudinal_quintic_coefficients(
                    start_s,
                    ego_vehicle.speed,
                    0.0,
                    target_longitudinal,
                    candidate_target_speed,
                    0.0,
                    duration,
                )
            elif lead_obstacle is not None and ego_state == "follow":
                lead_indices = lead_obstacle.indices_at(candidate_times)
                lead_s_at_end = float(lead_obstacle.longitudinal[lead_indices[-1]])
                lead_length_at_end = float(lead_obstacle.lengths[lead_indices[-1]])
                desired_gap = max(
                    self.follow_min_gap + self.follow_time_headway * ego_vehicle.speed,
                    ego_vehicle.LENGTH / 2
                    + lead_length_at_end / 2
                    + self.obstacle_margin,
                )
                lead_speed_at_end = float(lead_obstacle.speeds[lead_indices[-1]])
                candidate_target_speed = min(target_speed, lead_speed_at_end)
                target_longitudinal = np.clip(
                    lead_s_at_end - desired_gap,
                    start_s,
                    end_s,
                )
                longitudinal_coefficients = self._longitudinal_quintic_coefficients(
                    start_s,
                    ego_vehicle.speed,
                    0.0,
                    target_longitudinal,
                    candidate_target_speed,
                    0.0,
                    duration,
                )
            candidate_longitudinal = np.clip(
                self._evaluate(longitudinal_coefficients, candidate_times),
                start_s,
                end_s,
            )
            for target_lateral in lateral_targets:
                lateral_coefficients = self._quintic_coefficients(
                    lateral, float(target_lateral), duration
                )
                candidate_lateral = self._evaluate(
                    lateral_coefficients, candidate_times
                )
                candidate_points = self._positions_array(
                    current_lane, candidate_longitudinal, candidate_lateral
                )
                check_times = candidate_times
                check_points = candidate_points
                check_longitudinal = candidate_longitudinal
                check_lateral = candidate_lateral
                if self.footprint_check_substeps > 1 and len(candidate_times) > 1:
                    dense_times = np.linspace(
                        candidate_times[:-1, None],
                        candidate_times[1:, None],
                        self.footprint_check_substeps + 1,
                        axis=1,
                    ).reshape(-1)
                    check_times = np.unique(
                        np.concatenate((candidate_times[:1], dense_times))
                    )
                    check_longitudinal = np.clip(
                        self._evaluate(longitudinal_coefficients, check_times),
                        start_s,
                        end_s,
                    )
                    check_lateral = self._evaluate(lateral_coefficients, check_times)
                    check_points = self._positions_array(
                        current_lane, check_longitudinal, check_lateral
                    )
                candidate_lane_feasible = lane_feasible(
                    check_points,
                    check_times,
                    check_longitudinal,
                    check_lateral,
                )
                candidate_collision = not collision_free(
                    candidate_longitudinal,
                    candidate_lateral,
                    candidate_times,
                    nearby_obstacles,
                )
                candidate_cost = self._candidate_cost(
                    lateral_coefficients,
                    longitudinal_coefficients,
                    candidate_times,
                    float(target_lateral),
                    candidate_target_speed,
                )
                candidate = CandidateTrajectory(
                    points=candidate_points,
                    target_shift=float(target_lateral),
                    collision=candidate_collision,
                    lane_feasible=candidate_lane_feasible,
                    times=candidate_times,
                    longitudinal=candidate_longitudinal,
                    lateral=candidate_lateral,
                    cost=candidate_cost,
                    duration=duration,
                    target_speed=candidate_target_speed,
                )
                candidate_trajectories.append(candidate)
                if candidate_lane_feasible and not candidate_collision:
                    feasible_candidates.append((candidate_cost, candidate))

        collision_fallback = not feasible_candidates
        if feasible_candidates:
            _, selected = min(feasible_candidates, key=lambda item: item[0])
        else:
            selected = min(
                candidate_trajectories,
                key=lambda candidate: (
                    abs(candidate.target_shift),
                    abs(candidate.duration - self.trajectory_duration),
                ),
            )

        times = np.asarray(selected.times)
        longitudinal_samples = np.asarray(selected.longitudinal)
        lateral_samples = np.asarray(selected.lateral)
        target_shift = selected.target_shift
        points = self._positions_array(
            current_lane, longitudinal_samples, lateral_samples
        )
        velocities = np.gradient(points, times, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        headings = self._frenet_headings(
            current_lane,
            longitudinal_samples,
            lateral_samples,
            times,
            ego_vehicle.heading,
        )
        return LocalPlan(
            points,
            times,
            speeds,
            headings,
            longitudinal,
            lateral,
            target_shift,
            collision_fallback,
            tuple(candidate_trajectories),
        )
