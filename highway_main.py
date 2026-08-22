"""A minimal HighwayEnv closed-loop demo.

The default mode is headless and writes a screenshot to outputs/highway_snapshot.png.
Use --human to open the native pygame viewer instead.
"""

from __future__ import annotations

import argparse
import secrets
import time
from pathlib import Path

import gymnasium as gym
import highway_env
import matplotlib.pyplot as plt
import pygame
from highway_env.vehicle.behavior import IDMVehicle

from agents.ego_vehicle import EgoState, EgoVehicleAgent, GroundTruthDetector
from planning import (
    FixedLaneGlobalPlanner,
    LocalPlan,
    LocalPlanner,
)
from pipeline_factory import (
    create_pipeline,
    load_config,
    load_split_config,
)
from state_prediction import StatePredictor, create_state_predictor
from traffic_behavior import TrafficBehavior, create_traffic_behavior

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


def handle_target_lane_input(
    env: gym.Env,
    global_planner: FixedLaneGlobalPlanner,
    agent: EgoVehicleAgent | None,
) -> None:
    """Consume lane/state shortcuts and forward all other viewer events."""
    viewer = env.unwrapped.viewer
    if viewer is None:
        return

    for event in pygame.event.get():
        if event.type in {pygame.KEYDOWN, pygame.KEYUP} and event.key in {
            pygame.K_UP,
            pygame.K_DOWN,
            pygame.K_z,
            pygame.K_x,
            pygame.K_c,
        }:
            if event.type == pygame.KEYDOWN:
                state_keys = {
                    pygame.K_z: "cruise",
                    pygame.K_x: "follow",
                    pygame.K_c: "stop",
                }
                if event.key in state_keys:
                    if agent is not None:
                        agent.set_state(state_keys[event.key])
                else:
                    direction = -1 if event.key == pygame.K_UP else 1
                    changed = global_planner.shift_target_lane(
                        direction, env.unwrapped.vehicle
                    )
                    if changed and agent is not None:
                        agent.set_state("cruise")
            continue
        pygame.event.post(event)
    viewer.handle_events()


def draw_predictions(env: gym.Env, predictor: StatePredictor) -> None:
    """Overlay all NPC EKF forecasts on the HighwayEnv rendering surface."""
    highway = env.unwrapped
    viewer = highway.viewer
    if viewer is None:
        return

    surface = viewer.sim_surface
    horizon_steps = 25
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


def draw_frenet_reference(
    env: gym.Env,
    plan: LocalPlan,
    draw_all_trajectories: bool,
) -> None:
    """Draw all Frenet candidates and highlight the selected trajectory."""
    highway = env.unwrapped
    viewer = highway.viewer
    if viewer is None:
        return

    surface = viewer.sim_surface
    if draw_all_trajectories:
        for candidate in plan.candidates:
            candidate_pixels = [surface.vec2pix(point) for point in candidate.points]
            if len(candidate_pixels) <= 1:
                continue
            if candidate.collision:
                color = (230, 55, 55)
            elif not candidate.lane_feasible:
                color = (120, 120, 120)
            else:
                color = (80, 195, 115)
            pygame.draw.lines(surface, color, False, candidate_pixels, 1)

    pixels = [surface.vec2pix(point) for point in plan.points]
    if len(pixels) > 1:
        color = (235, 60, 60) if plan.collision_fallback else (255, 170, 40)
        pygame.draw.lines(surface, color, False, pixels, 3)


def draw_status_overlay(
    env: gym.Env,
    ego_state: EgoState | str,
    target_lane: int | None,
    physics_fps: float | None,
) -> None:
    """Draw ego speed, planning state, and effective physics rate."""
    highway = env.unwrapped
    viewer = highway.viewer
    if viewer is None:
        return

    surface = viewer.sim_surface
    vehicle = highway.vehicle
    speed_kph = float(vehicle.speed) * 3.6
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)
    panel = pygame.Surface((225, 52), pygame.SRCALPHA)
    panel.fill((12, 18, 24, 205))
    panel.blit(
        font.render(f"Speed  {speed_kph:5.1f} km/h", True, (245, 245, 245)),
        (10, 7),
    )
    panel.blit(
        small_font.render(
            f"State   {ego_state}    Lane   {target_lane if target_lane is not None else '-'}",
            True,
            (255, 195, 90),
        ),
        (10, 31),
    )
    surface.blit(panel, (10, 10))

    physics_text = (
        f"Physics {physics_fps:5.1f} Hz"
        if physics_fps is not None
        else "Physics -- Hz"
    )
    physics_text_surface = small_font.render(
        physics_text, True, (150, 220, 255)
    )
    physics_panel_width = physics_text_surface.get_width() + 20
    physics_panel_height = physics_text_surface.get_height() + 10
    physics_panel = pygame.Surface(
        (physics_panel_width, physics_panel_height), pygame.SRCALPHA
    )
    physics_panel.fill((12, 18, 24, 205))
    physics_panel.blit(physics_text_surface, (10, 5))
    surface.blit(
        physics_panel,
        (surface.get_width() - physics_panel_width - 10, 10),
    )


def render_with_predictions(
    env: gym.Env,
    predictor: StatePredictor,
    plan: LocalPlan,
    human: bool,
    draw_all_trajectories: bool,
    ego_state: EgoState | str = "cruise",
    target_lane: int | None = None,
    physics_fps: float | None = None,
) -> np.ndarray:
    highway = env.unwrapped
    viewer = highway.viewer
    if human and viewer is None:
        env.render()
        viewer = highway.viewer
    if human:
        viewer.offscreen = True
        viewer.display()
    else:
        env.render()
        viewer = highway.viewer
    draw_predictions(env, predictor)
    draw_frenet_reference(env, plan, draw_all_trajectories)
    draw_status_overlay(env, ego_state, target_lane, physics_fps)
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


def init_highwayenv(config:dict, render_mode:str, use_continuous_controller:bool):
    gym.register_envs(highway_env)

    if use_continuous_controller:
        config["action"] = {"type": "ContinuousAction"}
    
    env = gym.make(
        "highway-v0",
        render_mode=render_mode,
        config=config,
    )
    
    return env

def run_highwayenv(
    steps: int,
    seed: int | None,
    human: bool,
    output: Path,
    manual: bool,
    traffic_behavior: TrafficBehavior,
    ego_controller: str,
    state_predictor_name: str,
    draw_all_trajectories: bool,
    ego_state: EgoState,
    settings: dict,
) -> None:
    settings = dict(settings)
    config = dict(settings.get("environment", {}))
    config.setdefault("lanes_count", 3)
    config.setdefault("vehicles_count", 20)
    config.setdefault("duration", 120)
    config.setdefault("policy_frequency", 20)
    config.setdefault("simulation_frequency", 60)

    # Controller and planning state may be selected from the CLI. Planning
    # speed and initial target lane are always read from the Ego config.
    settings["ego"] = dict(settings.get("ego", {}))
    settings["ego"].update(
        controller=ego_controller,
        state=ego_state,
    )
    render_mode = "human" if human else "rgb_array"
    use_continuous_controller = ego_controller in {"pid", "pure_pursuit", "mpc", "nmpc"}
    
    # Initiating environment
    env = init_highwayenv(config = config, render_mode = render_mode, use_continuous_controller = use_continuous_controller)
    episode_seed = seed if seed is not None else secrets.randbits(32)
    observation, info = env.reset(seed=episode_seed)
    if not manual and not use_continuous_controller:
        install_idm_mobil_vehicle(env)    
    traffic_behavior.reset(npc_vehicles(env), env.unwrapped.np_random)
    
    # Initiating planner, predictor, controller
    pipeline = create_pipeline(settings, ego_controller)
    global_planner = pipeline.global_planner
    planner: LocalPlanner = pipeline.local_planner
    predictor: StatePredictor = create_state_predictor(
        state_predictor_name,
        settings,
    )
    controller = pipeline.controller
    agent = (
        EgoVehicleAgent(
            detector=GroundTruthDetector(),
            state_predictor=predictor,
            planner=planner,
            controller=controller,
            global_planner=global_planner,
        )
        if controller is not None
        else None
    )
    if agent:
        agent.reset(npc_vehicles(env))
        agent.set_state(ego_state)
    else:
        predictor.reset(npc_vehicles(env))
    
    total_reward = 0.0
    frame = None
    effective_physics_fps: float | None = None
    last_policy_start: float | None = None
    print(f"seed={episode_seed}")
    try:
        for step in range(steps):
            policy_start = time.perf_counter()
            if last_policy_start is not None:
                policy_cycle_elapsed = policy_start - last_policy_start
                physics_steps_per_policy = (
                    env.unwrapped.config["simulation_frequency"]
                    / env.unwrapped.config["policy_frequency"]
                )
                if policy_cycle_elapsed > 0.0:
                    measured_physics_fps = (
                        physics_steps_per_policy / policy_cycle_elapsed
                    )
                    effective_physics_fps = (
                        measured_physics_fps
                        if effective_physics_fps is None
                        else 0.85 * effective_physics_fps
                        + 0.15 * measured_physics_fps
                    )
            last_policy_start = policy_start
            policy_dt = 1 / env.unwrapped.config["policy_frequency"]
            if human:
                handle_target_lane_input(env, global_planner, agent)
            traffic_behavior.update(npc_vehicles(env), env.unwrapped.np_random, policy_dt)
            if agent:
                agent_step = agent.step(
                    env.unwrapped.vehicle,
                    npc_vehicles(env),
                    policy_dt,
                )
                control_plan = agent_step.plan
                action = agent_step.action
            elif manual:
                action = action_for_step(env, step)
            else:
                action = 0
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if agent:
                agent.update_obstacles(npc_vehicles(env), policy_dt)
            else:
                predictor.update(npc_vehicles(env), policy_dt)
                
            should_render = human or step == steps - 1 or terminated or truncated
            render_plan = None
            if should_render:
                render_plan = (
                    agent.plan(env.unwrapped.vehicle, npc_vehicles(env))
                    if agent
                    else planner.plan(env.unwrapped.vehicle, npc_vehicles(env))
                )

            if human:
                render_with_predictions(
                    env,
                    predictor,
                    render_plan,
                    human=True,
                    draw_all_trajectories=draw_all_trajectories,
                    ego_state=agent.state if agent else "cruise",
                    target_lane=global_planner.target_lane,
                    physics_fps=effective_physics_fps,
                )
            elif should_render:
                frame = render_with_predictions(
                    env,
                    predictor,
                    render_plan,
                    human=False,
                    draw_all_trajectories=draw_all_trajectories,
                    ego_state=agent.state if agent else "cruise",
                    target_lane=global_planner.target_lane,
                    physics_fps=effective_physics_fps,
                )
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
    parser.add_argument("--steps", type=int, default=5000, help="Maximum policy steps.")
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
        default="default",
        help="Background traffic behavior.",
    )
    parser.add_argument(
        "--ego-state",
        choices=("cruise", "follow", "stop"),
        default=None,
        help="Initial ego planning state; human mode can switch it with z/x/c.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("highway_snapshot.png"),
        help="Screenshot path in headless mode.",
    )
    parser.add_argument(
        "--state-predictor",
        choices=("ekf",),
        default=None,
        help="NPC trajectory prediction algorithm.",
    )
    parser.add_argument(
        "--draw-all-trajectories",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw all Frenet candidates in addition to the selected trajectory.",
    )
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=Path(__file__).with_name("configs") / "environment_config.jsonc",
        help="JSON/JSONC file containing only HighwayEnv parameters.",
    )
    parser.add_argument(
        "--ego-config",
        type=Path,
        default=Path(__file__).with_name("configs") / "ego_config.jsonc",
        help="JSON/JSONC file containing the complete Ego pipeline configuration.",
    )
    args = parser.parse_args()

    config = load_split_config(args.environment_config, args.ego_config)
    ego_config = config.get("ego", {})
    
    args.ego_controller = ego_config.get(
        "controller", "pure_pursuit"
    )
    args.ego_state = args.ego_state or ego_config.get("state", "cruise")

    args.state_predictor = args.state_predictor or ego_config.get(
        "state_predictor", "ekf"
    )
    args.pipeline_config = config
    if args.manual and args.ego_controller != "idm":
        parser.error("--manual can only be used with --ego-controller idm")
    return args


if __name__ == "__main__":
    args = parse_args()
    traffic_behavior = create_traffic_behavior(args.traffic_behavior)
    run_highwayenv(
        args.steps,
        args.seed,
        args.human,
        args.output,
        args.manual,
        traffic_behavior,
        args.ego_controller,
        args.state_predictor,
        args.draw_all_trajectories,
        args.ego_state,
        args.pipeline_config,
    )
