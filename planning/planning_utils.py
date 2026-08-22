"""Pure numerical and geometry helpers shared by planning algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np


def quintic_coefficients(
    start: float,
    end: float,
    duration: float,
    start_velocity: float = 0.0,
    start_acceleration: float = 0.0,
    end_velocity: float = 0.0,
    end_acceleration: float = 0.0,
) -> np.ndarray:
    """Fit a quintic polynomial to position, velocity, and acceleration bounds."""
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
    return np.array([start, start_velocity, 0.5 * start_acceleration, *tail])


def quartic_coefficients(
    start: float,
    start_speed: float,
    target_speed: float,
    duration: float,
) -> np.ndarray:
    """Fit longitudinal position with endpoint speed and acceleration bounds."""
    if duration <= 0.0:
        raise ValueError("Quartic duration must be positive.")
    matrix = np.array(
        [
            [3 * duration**2, 4 * duration**3],
            [6 * duration, 12 * duration**2],
        ]
    )
    tail = np.linalg.solve(matrix, np.array([target_speed - start_speed, 0.0]))
    return np.array([start, start_speed, 0.0, *tail])


def longitudinal_quintic_coefficients(
    start: float,
    start_speed: float,
    start_acceleration: float,
    end: float,
    end_speed: float,
    end_acceleration: float,
    duration: float,
) -> np.ndarray:
    """Fit longitudinal position to terminal position, speed, and acceleration."""
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
                end - start - start_speed * duration
                - 0.5 * start_acceleration * duration**2,
                end_speed - start_speed - start_acceleration * duration,
                end_acceleration - start_acceleration,
            ]
        ),
    )
    return np.array([start, start_speed, 0.5 * start_acceleration, *tail])


def evaluate_polynomial(coefficients: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Evaluate a polynomial represented by ascending power coefficients."""
    return sum(
        coefficient * times**power
        for power, coefficient in enumerate(coefficients)
    )


def local_coordinates_array(lane: Any, positions: np.ndarray) -> np.ndarray:
    """Convert positions to lane coordinates, using vectorized lane geometry when possible."""
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
        radius = np.linalg.norm(delta, axis=1)
        longitudinal = lane.direction * phase_delta * lane.radius
        lateral = lane.direction * (lane.radius - radius)
        return np.column_stack((longitudinal, lateral))
    return np.asarray([lane.local_coordinates(position) for position in positions])


def positions_array(
    lane: Any,
    longitudinal: np.ndarray,
    lateral: np.ndarray,
) -> np.ndarray:
    """Convert lane coordinates to positions, vectorizing common lane types."""
    longitudinal = np.asarray(longitudinal, dtype=float)
    lateral = np.asarray(lateral, dtype=float)
    if all(
        hasattr(lane, attribute)
        for attribute in ("start", "direction", "direction_lateral")
    ):
        return (
            np.asarray(lane.start, dtype=float)
            + longitudinal[:, None] * np.asarray(lane.direction, dtype=float)
            + lateral[:, None] * np.asarray(lane.direction_lateral, dtype=float)
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


def sample_times(duration: float, dt: float) -> np.ndarray:
    """Sample a trajectory interval while always retaining its exact endpoint."""
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("Trajectory duration and dt must be positive.")
    step_count = int(np.floor(duration / dt + 1e-9))
    times = np.arange(step_count + 1, dtype=float) * dt
    if times[-1] < duration - 1e-9:
        times = np.append(times, duration)
    else:
        times[-1] = duration
    return times


def derivative(
    coefficients: np.ndarray,
    times: np.ndarray,
    order: int,
) -> np.ndarray:
    """Evaluate a polynomial derivative of the requested order."""
    derivative_coefficients = coefficients.copy()
    for _ in range(order):
        derivative_coefficients = np.array(
            [power * value for power, value in enumerate(derivative_coefficients)][1:]
        )
    if len(derivative_coefficients) == 0:
        return np.zeros_like(times)
    return evaluate_polynomial(derivative_coefficients, times)


def candidate_cost(
    lateral_coefficients: np.ndarray,
    longitudinal_coefficients: np.ndarray,
    times: np.ndarray,
    target_lateral: float,
    target_speed: float,
    trajectory_dt: float,
    target_speed_mps: float,
) -> float:
    """Score smoothness, duration, lateral offset, and terminal speed error."""
    lateral_jerk = derivative(lateral_coefficients, times, 3)
    longitudinal_jerk = derivative(longitudinal_coefficients, times, 3)
    return float(
        0.05 * np.sum(lateral_jerk**2) * trajectory_dt
        + 0.005 * np.sum(longitudinal_jerk**2) * trajectory_dt
        + 0.1 * times[-1]
        + 0.5 * target_lateral**2
        + 0.02 * (target_speed - target_speed_mps) ** 2
    )


def frenet_headings(
    lane: Any,
    longitudinal: np.ndarray,
    lateral: np.ndarray,
    times: np.ndarray,
    fallback_heading: float,
) -> np.ndarray:
    """Compute headings from lane tangent and relative Frenet velocity."""
    longitudinal = np.asarray(longitudinal, dtype=float)
    lateral = np.asarray(lateral, dtype=float)
    times = np.asarray(times, dtype=float)
    longitudinal_speed = np.gradient(longitudinal, times)
    lateral_speed = np.gradient(lateral, times)
    relative_headings = np.arctan2(lateral_speed, longitudinal_speed)
    valid = np.abs(longitudinal_speed) > 1e-6
    for index in range(len(relative_headings)):
        if not valid[index]:
            relative_headings[index] = (
                relative_headings[index - 1] if index > 0 else 0.0
            )
    lane_headings = np.asarray(
        [
            lane.heading_at(float(np.clip(sample, 0.0, lane.length)))
            for sample in longitudinal
        ],
        dtype=float,
    )
    headings = lane_headings + relative_headings
    if len(headings) and not np.isfinite(headings[0]):
        headings[0] = fallback_heading
    return headings


__all__ = [
    "candidate_cost",
    "derivative",
    "evaluate_polynomial",
    "frenet_headings",
    "local_coordinates_array",
    "longitudinal_quintic_coefficients",
    "positions_array",
    "quartic_coefficients",
    "quintic_coefficients",
    "sample_times",
]
