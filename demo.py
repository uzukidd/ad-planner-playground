"""A minimal HighwayEnv closed-loop demo.

The default mode is headless and writes a screenshot to outputs/highway_snapshot.png.
Use --human to open the native pygame viewer instead.
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import pygame
from highway_env.vehicle.behavior import IDMVehicle

from local_planner import FrenetLocalPlanner, LocalPlan, LocalPlanner
from state_predictor import EKFStatePredictor, StatePredictor
from traffic_behavior import TrafficBehavior, create_traffic_behavior
from trajectory_controller import (
    MPCTrajectoryController,
    NMPCTrajectoryController,
    PIDTrajectoryController,
    TrajectoryController,
)

class OvertakingVehicle(IDMVehicle):
    """Stable IDM/MOBIL ego vehicle that overtakes only when it is worthwhile."""

    POLITENESS = 0.35
    LANE_CHANGE_MIN_ACC_GAIN = 0.3
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.0
    LANE_CHANGE_DELAY = 2.0
    TIME_WANTED = 1.8
    DISTANCE_WANTED = 10.0
    OVERTAKE_DISTANCE = 45.0

    def change_lane_policy(self) -> None:
        """Try an available safe lane when a slower vehicle blocks the way."""
        if self.lane_index != self.target_lane_index:
            return super().change_lane_policy()

        front_vehicle, _ = self.road.neighbour_vehicles(self, self.lane_index)
        blocked = (
            front_vehicle is not None
            and 0 < self.lane_distance_to(front_vehicle) < self.OVERTAKE_DISTANCE
            and front_vehicle.speed < self.target_speed - 1.0
        )
        if blocked and self.timer >= self.LANE_CHANGE_DELAY:
            for lane_index in self.road.network.side_lanes(self.lane_index):
                lane = self.road.network.get_lane(lane_index)
                if lane.is_reachable_from(self.position) and self.mobil(lane_index):
                    self.target_lane_index = lane_index
                    self.timer = 0
                    return

        super().change_lane_policy()


def npc_vehicles(env: gym.Env) -> list[IDMVehicle]:
    highway = env.unwrapped
    controlled = set(highway.controlled_vehicles)
    return [vehicle for vehicle in highway.road.vehicles if vehicle not in controlled]


def draw_predictions(env: gym.Env, predictor: StatePredictor) -> None:
    """Overlay all NPC EKF forecasts on the HighwayEnv rendering surface."""
    highway = env.unwrapped
    viewer = highway.viewer
    if viewer is None:
        return

    surface = viewer.sim_surface
    horizon_steps = 100
    prediction_dt = 1 / highway.config["policy_frequency"]
    for trajectory in predictor.predict_trajectories(
        horizon_steps, prediction_dt
    ).values():
        pixels = [surface.vec2pix(position) for position in trajectory]
        if len(pixels) > 1:
            pygame.draw.lines(surface, (40, 220, 210), False, pixels, 2)
        for index, pixel in enumerate(pixels[::2]):
            radius = max(2, 5 - index // 3)
            pygame.draw.circle(surface, (160, 255, 235), pixel, radius)


def draw_frenet_reference(env: gym.Env, plan: LocalPlan) -> None:
    """Draw the current-lane Frenet reference path without affecting control."""
    highway = env.unwrapped
    viewer = highway.viewer
    if viewer is None:
        return

    pixels = [viewer.sim_surface.vec2pix(point) for point in plan.points]
    if len(pixels) > 1:
        color = (235, 60, 60) if plan.collision_fallback else (255, 170, 40)
        pygame.draw.lines(viewer.sim_surface, color, False, pixels, 3)


def render_with_predictions(
    env: gym.Env,
    predictor: StatePredictor,
    plan: LocalPlan,
    human: bool,
) -> np.ndarray:
    env.render()
    draw_predictions(env, predictor)
    draw_frenet_reference(env, plan)
    viewer = env.unwrapped.viewer
    if human:
        viewer.screen.blit(viewer.sim_surface, (0, 0))
        pygame.display.flip()
    return viewer.get_image()


def install_idm_mobil_vehicle(env: gym.Env) -> None:
    """Replace the controlled vehicle with IDM longitudinal and MOBIL lateral control."""
    highway = env.unwrapped
    old_vehicle = highway.controlled_vehicles[0]
    new_vehicle = OvertakingVehicle(
        highway.road,
        old_vehicle.position.copy(),
        heading=old_vehicle.heading,
        speed=old_vehicle.speed,
    )
    new_vehicle.target_lane_index = old_vehicle.target_lane_index
    if hasattr(old_vehicle, "color"):
        new_vehicle.color = old_vehicle.color

    highway.road.vehicles[highway.road.vehicles.index(old_vehicle)] = new_vehicle
    highway.controlled_vehicles[0] = new_vehicle


def run_demo(
    steps: int,
    seed: int | None,
    human: bool,
    output: Path,
    manual: bool,
    traffic_behavior: TrafficBehavior,
    ego_controller: str,
    target_speed: float,
) -> None:
    gym.register_envs(highway_env)
    render_mode = "human" if human else "rgb_array"
    use_continuous_controller = ego_controller in {"pid", "mpc", "nmpc"}
    config = {
        "lanes_count": 3,
        "vehicles_count": 10,
        "duration": 120,
        "policy_frequency": 20,
        "simulation_frequency": 60,
    }
    if use_continuous_controller:
        config["action"] = {"type": "ContinuousAction"}
    env = gym.make(
        "highway-v0",
        render_mode=render_mode,
        config=config,
    )

    episode_seed = seed if seed is not None else secrets.randbits(32)
    observation, info = env.reset(seed=episode_seed)
    if not manual and not use_continuous_controller:
        install_idm_mobil_vehicle(env)
    traffic_behavior.reset(npc_vehicles(env), env.unwrapped.np_random)
    planner: LocalPlanner = FrenetLocalPlanner(target_speed=target_speed)
    predictor: StatePredictor = EKFStatePredictor()
    controller: TrajectoryController | None = {
        "pid": PIDTrajectoryController,
        "mpc": MPCTrajectoryController,
        "nmpc": NMPCTrajectoryController,
    }.get(ego_controller, lambda: None)()
    if controller:
        controller.reset()
    predictor.reset(npc_vehicles(env))
    total_reward = 0.0
    frame = None
    print(f"seed={episode_seed}")
    try:
        for step in range(steps):
            policy_dt = 1 / env.unwrapped.config["policy_frequency"]
            traffic_behavior.update(npc_vehicles(env), env.unwrapped.np_random, policy_dt)
            control_plan = planner.plan(env.unwrapped.vehicle, npc_vehicles(env))
            if controller:
                action = controller.action(
                    env.unwrapped.vehicle, control_plan, policy_dt
                )
            elif manual:
                action = action_for_step(env, step)
            else:
                action = 0
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            predictor.update(npc_vehicles(env), policy_dt)
            render_plan = planner.plan(env.unwrapped.vehicle, npc_vehicles(env))

            if human:
                render_with_predictions(env, predictor, render_plan, human=True)
            elif step == steps - 1 or terminated or truncated:
                frame = render_with_predictions(env, predictor, render_plan, human=False)
            if terminated or truncated:
                break
    finally:
        env.close()

    print(f"steps={step + 1}, total_reward={total_reward:.3f}")
    print(f"observation_shape={observation.shape}")

    if frame is not None:
        if not output.parent.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(output, frame)
        print(f"snapshot={output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal HighwayEnv demo.")
    parser.add_argument("--steps", type=int, default=10000, help="Maximum policy steps.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional fixed seed; omit it to generate a new seed each run.",
    )
    parser.add_argument("--human", action="store_true", help="Open the pygame viewer.")
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Use the original hand-written policy instead of IDM/MOBIL.",
    )
    parser.add_argument(
        "--traffic-behavior",
        choices=("default", "random"),
        default="random",
        help="Background traffic behavior.",
    )
    parser.add_argument(
        "--ego-controller",
        choices=("idm", "pid", "mpc", "nmpc"),
        default="idm",
        help="Ego driving controller.",
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=70.0,
        help="Ego target speed in m/s after an obstacle clears.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("highway_snapshot.png"),
        help="Screenshot path in headless mode.",
    )
    args = parser.parse_args()
    if args.manual and args.ego_controller != "idm":
        parser.error("--manual can only be used with --ego-controller idm")
    return args


if __name__ == "__main__":
    args = parse_args()
    traffic_behavior = create_traffic_behavior(args.traffic_behavior)
    run_demo(
        args.steps,
        args.seed,
        args.human,
        args.output,
        args.manual,
        traffic_behavior,
        args.ego_controller,
        args.target_speed,
    )
