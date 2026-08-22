"""Pluggable local-planning interfaces and a Frenet implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from ..global_planning.global_planner import GlobalPlan
from ..planning_utils import (
    candidate_cost as candidate_cost_fn,
    evaluate_polynomial,
    frenet_headings,
    local_coordinates_array,
    longitudinal_quintic_coefficients,
    positions_array,
    quartic_coefficients,
    quintic_coefficients,
    sample_times,
)


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

    def prediction_times(self, ego_state: str = "cruise") -> np.ndarray: ...


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
    state_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert the configured target speed from km/h to internal m/s."""
        if not np.isfinite(self.target_speed) or self.target_speed < 0.0:
            raise ValueError("target_speed must be a finite nonnegative value in km/h.")
        allowed_states = {"cruise", "follow", "stop"}
        unknown_states = set(self.state_configs) - allowed_states
        if unknown_states:
            names = ", ".join(sorted(unknown_states))
            raise ValueError(f"Unknown local-planner state configuration(s): {names}")
        for state, parameters in self.state_configs.items():
            if not isinstance(parameters, dict):
                raise ValueError(
                    f"local_planner.state_configs.{state} must be an object."
                )
            supported_parameters = {
                "trajectory_duration",
                "trajectory_dt",
                "maneuver_duration",
                "target_speed",
                "duration_samples",
                "obstacle_margin",
                "lane_boundary_tolerance",
                "footprint_check_substeps",
                "follow_time_headway",
                "follow_min_gap",
                "stop_gap",
            }
            unknown_parameters = set(parameters) - supported_parameters
            if unknown_parameters:
                names = ", ".join(sorted(unknown_parameters))
                raise ValueError(
                    f"Unknown state_configs.{state} parameter(s): {names}"
                )
            if "duration_samples" in parameters:
                parameters["duration_samples"] = tuple(parameters["duration_samples"])
            if "target_speed" in parameters:
                speed = parameters["target_speed"]
                if not np.isfinite(speed) or speed < 0.0:
                    raise ValueError(
                        f"state_configs.{state}.target_speed must be finite and nonnegative."
                    )
        self.target_speed_mps = float(self.target_speed) / 3.6

    def _state_parameters(self, ego_state: str) -> dict[str, Any]:
        """Resolve state-specific planner and sampling values."""
        if ego_state not in {"cruise", "follow", "stop"}:
            raise ValueError(f"Unsupported ego state: {ego_state}")
        parameters: dict[str, Any] = {
            "trajectory_duration": self.trajectory_duration,
            "trajectory_dt": self.trajectory_dt,
            "maneuver_duration": self.maneuver_duration,
            "target_speed": self.target_speed,
            "target_speed_mps": self.target_speed_mps,
            "duration_samples": self.duration_samples,
            "obstacle_margin": self.obstacle_margin,
            "lane_boundary_tolerance": self.lane_boundary_tolerance,
            "footprint_check_substeps": self.footprint_check_substeps,
            "follow_time_headway": self.follow_time_headway,
            "follow_min_gap": self.follow_min_gap,
            "stop_gap": self.stop_gap,
        }
        parameters.update(self.state_configs.get(ego_state, {}))
        parameters["duration_samples"] = tuple(parameters["duration_samples"])
        parameters["target_speed_mps"] = float(parameters["target_speed"]) / 3.6
        return parameters

    def prediction_times(self, ego_state: str = "cruise") -> np.ndarray:
        """Timestamps shared by Ego planning and obstacle prediction."""
        parameters = self._state_parameters(ego_state)
        maximum_duration = max(self._candidate_durations(parameters))
        return sample_times(maximum_duration, parameters["trajectory_dt"])

    def _candidate_durations(
        self, parameters: dict[str, Any] | None = None
    ) -> tuple[float, ...]:
        """Return sorted positive terminal times, including the legacy value."""
        parameters = parameters or self._state_parameters("cruise")
        durations = {float(parameters["trajectory_duration"])}
        durations.update(float(duration) for duration in parameters["duration_samples"])
        return tuple(sorted(duration for duration in durations if duration > 0.0))

    def plan(
        self,
        ego_vehicle: Any,
        obstacles: Sequence[Any],
        ego_state: str = "cruise",
        reference_route: GlobalPlan | None = None,
    ) -> LocalPlan:
        if ego_state not in {"cruise", "follow", "stop"}:
            raise ValueError(f"Unsupported ego state: {ego_state}")
        parameters = self._state_parameters(ego_state)
        trajectory_dt = float(parameters["trajectory_dt"])
        trajectory_duration = float(parameters["trajectory_duration"])
        target_speed_mps = float(parameters["target_speed_mps"])
        obstacle_margin = float(parameters["obstacle_margin"])
        lane_boundary_tolerance = float(parameters["lane_boundary_tolerance"])
        footprint_check_substeps = int(parameters["footprint_check_substeps"])
        follow_time_headway = float(parameters["follow_time_headway"])
        follow_min_gap = float(parameters["follow_min_gap"])
        stop_gap = float(parameters["stop_gap"])
        
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
        target_speed = target_speed_mps
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
            frenet_coordinates = local_coordinates_array(
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
                    + obstacle_margin
                )
                lateral_clearance = (
                    ego_vehicle.WIDTH / 2
                    + obstacle.widths[indices] / 2
                    + obstacle_margin
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
            headings = frenet_headings(
                current_lane,
                candidate_longitudinal,
                candidate_lateral,
                candidate_times,
                ego_vehicle.heading,
            )

            half_length = float(ego_vehicle.LENGTH) / 2.0
            half_width = float(ego_vehicle.WIDTH) / 2.0
            tolerance = lane_boundary_tolerance
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
                coordinates = local_coordinates_array(
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
        for duration in self._candidate_durations(parameters):
            candidate_times = sample_times(duration, trajectory_dt)
            longitudinal_coefficients = quartic_coefficients(
                start_s, ego_vehicle.speed, target_speed, duration
            )
            candidate_target_speed = target_speed
            if ego_state == "stop":
                if lead_obstacle is not None:
                    lead_indices = lead_obstacle.indices_at(candidate_times)
                    lead_s_at_end = float(lead_obstacle.longitudinal[lead_indices[-1]])
                    lead_length_at_end = float(lead_obstacle.lengths[lead_indices[-1]])
                    desired_gap = max(
                        stop_gap,
                        ego_vehicle.LENGTH / 2
                        + lead_length_at_end / 2
                        + obstacle_margin,
                    )
                    target_longitudinal = np.clip(
                        lead_s_at_end - desired_gap,
                        start_s,
                        end_s,
                    )
                else:
                    braking_acceleration = max(1.0, 0.5 * obstacle_margin + 3.0)
                    stopping_distance = ego_vehicle.speed**2 / (2.0 * braking_acceleration)
                    target_longitudinal = np.clip(
                        start_s + stopping_distance,
                        start_s,
                        end_s,
                    )
                candidate_target_speed = 0.0
                longitudinal_coefficients = longitudinal_quintic_coefficients(
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
                    follow_min_gap + follow_time_headway * ego_vehicle.speed,
                    ego_vehicle.LENGTH / 2
                    + lead_length_at_end / 2
                    + obstacle_margin,
                )
                lead_speed_at_end = float(lead_obstacle.speeds[lead_indices[-1]])
                candidate_target_speed = min(target_speed, lead_speed_at_end)
                target_longitudinal = np.clip(
                    lead_s_at_end - desired_gap,
                    start_s,
                    end_s,
                )
                longitudinal_coefficients = longitudinal_quintic_coefficients(
                    start_s,
                    ego_vehicle.speed,
                    0.0,
                    target_longitudinal,
                    candidate_target_speed,
                    0.0,
                    duration,
                )
            candidate_longitudinal = np.clip(
                evaluate_polynomial(longitudinal_coefficients, candidate_times),
                start_s,
                end_s,
            )
            for target_lateral in lateral_targets:
                lateral_coefficients = quintic_coefficients(
                    lateral, float(target_lateral), duration
                )
                candidate_lateral = evaluate_polynomial(
                    lateral_coefficients, candidate_times
                )
                candidate_points = positions_array(
                    current_lane, candidate_longitudinal, candidate_lateral
                )
                check_times = candidate_times
                check_points = candidate_points
                check_longitudinal = candidate_longitudinal
                check_lateral = candidate_lateral
                if footprint_check_substeps > 1 and len(candidate_times) > 1:
                    dense_times = np.linspace(
                        candidate_times[:-1, None],
                        candidate_times[1:, None],
                        footprint_check_substeps + 1,
                        axis=1,
                    ).reshape(-1)
                    check_times = np.unique(
                        np.concatenate((candidate_times[:1], dense_times))
                    )
                    check_longitudinal = np.clip(
                        evaluate_polynomial(longitudinal_coefficients, check_times),
                        start_s,
                        end_s,
                    )
                    check_lateral = evaluate_polynomial(lateral_coefficients, check_times)
                    check_points = positions_array(
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
                candidate_cost = candidate_cost_fn(
                    lateral_coefficients,
                    longitudinal_coefficients,
                    candidate_times,
                    float(target_lateral),
                    candidate_target_speed,
                    trajectory_dt,
                    target_speed_mps,
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
                    abs(candidate.duration - trajectory_duration),
                ),
            )

        times = np.asarray(selected.times)
        longitudinal_samples = np.asarray(selected.longitudinal)
        lateral_samples = np.asarray(selected.lateral)
        target_shift = selected.target_shift
        points = positions_array(
            current_lane, longitudinal_samples, lateral_samples
        )
        velocities = np.gradient(points, times, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        headings = frenet_headings(
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
