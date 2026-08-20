"""Composable detector -> planner -> controller pipeline for the ego vehicle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

import numpy as np

from control import TrajectoryController
from planning import GlobalPlan, GlobalPlanner, LocalPlan, LocalPlanner
from state_prediction import ObstacleStatePredictor, TimedObstacleTrajectory


EgoState = Literal["cruise", "follow", "stop"]


@dataclass(frozen=True)
class DetectionResult:
    """Detector output consumed by the planner."""

    ego_vehicle: Any
    obstacles: tuple[Any, ...]


class Detector(Protocol):
    """Interface for replacing ground truth with sensors or perception later."""

    def detect(
        self, ego_vehicle: Any, obstacles: Sequence[Any]
    ) -> DetectionResult: ...


@dataclass
class GroundTruthDetector:
    """Expose HighwayEnv's current vehicle objects as perfect detections."""

    def detect(
        self, ego_vehicle: Any, obstacles: Sequence[Any]
    ) -> DetectionResult:
        return DetectionResult(ego_vehicle, tuple(obstacles))


@dataclass(frozen=True)
class AgentStep:
    """One complete detector -> planner -> controller pipeline result."""

    detection: DetectionResult
    obstacle_states: tuple[TimedObstacleTrajectory, ...]
    global_plan: GlobalPlan | None
    plan: LocalPlan
    action: np.ndarray
    ego_state: EgoState


@dataclass
class EgoVehicleAgent:
    """Autonomous-driving pipeline for one ego vehicle."""

    detector: Detector
    state_predictor: ObstacleStatePredictor
    planner: LocalPlanner
    controller: TrajectoryController
    global_planner: GlobalPlanner | None = None
    state: EgoState = "cruise"
    last_global_plan: GlobalPlan | None = field(default=None, init=False)
    last_step: AgentStep | None = field(default=None, init=False)

    def reset(self, obstacles: Sequence[Any] = ()) -> None:
        """Reset state held by the controller and clear the previous output."""
        self.controller.reset()
        self.state_predictor.reset(obstacles)
        self.state = "cruise"
        self.last_global_plan = None
        self.last_step = None

    def set_state(self, state: EgoState) -> None:
        """Select the explicit planning state used by the local planner."""
        if state not in {"cruise", "follow", "stop"}:
            raise ValueError(f"Unsupported ego state: {state}")
        self.state = state

    def update_obstacles(self, obstacles: Sequence[Any], dt: float) -> None:
        """Update predictor state once after a new simulator observation."""
        self.state_predictor.update(obstacles, dt)

    def detect_and_plan(
        self, ego_vehicle: Any, obstacles: Sequence[Any]
    ) -> tuple[
        DetectionResult,
        tuple[TimedObstacleTrajectory, ...],
        GlobalPlan | None,
        LocalPlan,
    ]:
        detection = self.detector.detect(ego_vehicle, obstacles)
        obstacle_states = self.state_predictor.predict(
            detection.obstacles,
            self.planner.prediction_times(),
        )
        global_plan = (
            self.global_planner.plan(detection.ego_vehicle)
            if self.global_planner is not None
            else None
        )
        plan = self.planner.plan(
            detection.ego_vehicle,
            obstacle_states,
            ego_state=self.state,
            reference_route=global_plan,
        )
        self.last_global_plan = global_plan
        return detection, obstacle_states, global_plan, plan

    def plan(self, ego_vehicle: Any, obstacles: Sequence[Any]) -> LocalPlan:
        """Generate a current plan for rendering or inspection without control."""
        _, _, _, plan = self.detect_and_plan(ego_vehicle, obstacles)
        return plan

    def step(
        self, ego_vehicle: Any, obstacles: Sequence[Any], dt: float
    ) -> AgentStep:
        """Run detection, planning, and control once and return the action."""
        detection, obstacle_states, global_plan, plan = self.detect_and_plan(
            ego_vehicle, obstacles
        )
        action = np.asarray(
            self.controller.action(detection.ego_vehicle, plan, dt),
            dtype=np.float32,
        )
        result = AgentStep(
            detection,
            obstacle_states,
            global_plan,
            plan,
            action,
            self.state,
        )
        self.last_step = result
        return result
