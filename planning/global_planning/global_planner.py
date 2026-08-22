"""Pluggable global planning interfaces and a fixed target-lane planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class GlobalPlan:
    """A lane-level reference route consumed by the local planner."""

    target_lane: int
    lane_index: tuple[str, str, int]
    lane: Any
    longitudinal: np.ndarray
    points: np.ndarray


class GlobalPlanner(Protocol):
    """Interface for replaceable lane-level route planners."""

    def plan(self, ego_vehicle: Any) -> GlobalPlan: ...


@dataclass
class FixedLaneGlobalPlanner:
    """Generate a reference route for a target changed only by explicit commands."""

    target_lane: int = 1
    horizon: float = 250.0
    point_spacing: float = 5.0

    @staticmethod
    def legal_target_lanes(ego_vehicle: Any) -> tuple[int, ...]:
        """Return sorted lane numbers available on the current road segment."""
        lane_indices = ego_vehicle.road.network.all_side_lanes(
            ego_vehicle.lane_index
        )
        return tuple(sorted({int(lane_index[2]) for lane_index in lane_indices}))

    def shift_target_lane(self, delta: int, ego_vehicle: Any) -> bool:
        """Move by one legal lane and clamp at the road boundaries."""
        if delta == 0:
            return False
        legal_lanes = self.legal_target_lanes(ego_vehicle)
        if not legal_lanes:
            return False
        if self.target_lane in legal_lanes:
            current_index = legal_lanes.index(self.target_lane)
        else:
            current_index = min(
                range(len(legal_lanes)),
                key=lambda index: abs(legal_lanes[index] - self.target_lane),
            )
        direction = -1 if delta < 0 else 1
        target_index = int(
            np.clip(current_index + direction, 0, len(legal_lanes) - 1)
        )
        next_lane = legal_lanes[target_index]
        changed = next_lane != self.target_lane
        self.target_lane = next_lane
        return changed

    def _lane_index(self, ego_vehicle: Any) -> tuple[str, str, int]:
        if self.target_lane < 0:
            raise ValueError("target_lane must be nonnegative.")
        lane_from, lane_to, _ = ego_vehicle.lane_index
        lane_index = (lane_from, lane_to, self.target_lane)
        try:
            ego_vehicle.road.network.get_lane(lane_index)
        except (IndexError, KeyError) as error:
            raise ValueError(
                f"Target lane {self.target_lane} is not available on "
                f"road segment {(lane_from, lane_to)}."
            ) from error
        return lane_index

    def plan(self, ego_vehicle: Any) -> GlobalPlan:
        """Sample the fixed target lane centerline ahead of the ego vehicle."""
        if self.horizon <= 0.0 or self.point_spacing <= 0.0:
            raise ValueError("Global planning horizon and spacing must be positive.")

        lane_index = self._lane_index(ego_vehicle)
        lane = ego_vehicle.road.network.get_lane(lane_index)
        start_s, _ = lane.local_coordinates(ego_vehicle.position)
        start_s = float(np.clip(start_s, 0.0, lane.length))
        end_s = float(min(lane.length, start_s + self.horizon))
        longitudinal = np.arange(start_s, end_s, self.point_spacing, dtype=float)
        if len(longitudinal) == 0 or longitudinal[-1] < end_s:
            longitudinal = np.append(longitudinal, end_s)
        points = np.asarray(
            [lane.position(float(sample_s), 0.0) for sample_s in longitudinal],
            dtype=float,
        )
        return GlobalPlan(
            target_lane=self.target_lane,
            lane_index=lane_index,
            lane=lane,
            longitudinal=longitudinal,
            points=points,
        )
