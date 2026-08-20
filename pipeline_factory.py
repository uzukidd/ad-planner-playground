"""Configuration loading and construction for the planning pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar

from control import (
    PIDTrajectoryController,
    PurePursuitTrajectoryController,
    TrajectoryController,
)
from planning import FixedLaneGlobalPlanner, FrenetLocalPlanner, GlobalPlanner, LocalPlanner


Config = Mapping[str, Any]
T = TypeVar("T")

_CONTROLLER_TYPES = {
    "pid": PIDTrajectoryController,
    "pure_pursuit": PurePursuitTrajectoryController,
}


@dataclass(frozen=True)
class PipelineComponents:
    """All configurable planning components used by ``EgoVehicleAgent``."""

    global_planner: GlobalPlanner
    local_planner: LocalPlanner
    controller: TrajectoryController | None


def _strip_jsonc_comments(text: str) -> str:
    """Remove // comments without changing comment-like text inside strings."""
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue
        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        if character == "/" and index + 1 < len(text) and text[index + 1] == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        output.append(character)
        index += 1
    return "".join(output)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or JSONC configuration file and validate its root."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    config = json.loads(
        _strip_jsonc_comments(text) if config_path.suffix.lower() == ".jsonc" else text
    )
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be an object: {config_path}")
    return config


def load_split_config(
    environment_path: str | Path,
    ego_path: str | Path,
) -> dict[str, Any]:
    """Load separate environment and Ego-pipeline configuration files."""
    environment = load_config(environment_path)
    ego_pipeline = load_config(ego_path)
    if "environment" in environment:
        raise ValueError(
            "The split environment config must contain environment parameters at its root."
        )
    if "environment" in ego_pipeline:
        raise ValueError(
            "The Ego config must not contain an environment section."
        )
    return {"environment": environment, **ego_pipeline}


def _dataclass_from_config(component_type: type[T], params: Mapping[str, Any]) -> T:
    """Construct a dataclass while catching misspelled configuration keys."""
    valid_fields = {
        field.name
        for field in fields(component_type)
        if field.init and not field.name.startswith("_")
    }
    unknown_fields = set(params) - valid_fields
    if unknown_fields:
        names = ", ".join(sorted(unknown_fields))
        raise ValueError(f"Unknown {component_type.__name__} parameter(s): {names}")

    values = dict(params)
    if "duration_samples" in values:
        values["duration_samples"] = tuple(values["duration_samples"])
    return component_type(**values)


def create_global_planner(config: Config) -> GlobalPlanner:
    """Create the configured global planner."""
    params = config.get("global_planner", {})
    return _dataclass_from_config(FixedLaneGlobalPlanner, params)


def create_local_planner(config: Config) -> LocalPlanner:
    """Create the configured local planner."""
    params = config.get("local_planner", {})
    return _dataclass_from_config(FrenetLocalPlanner, params)


def create_controller(
    config: Config,
    name: str | None = None,
) -> TrajectoryController | None:
    """Create a trajectory controller, or ``None`` for IDM mode."""
    ego_config = config.get("ego", {})
    controller_name = name or ego_config.get("controller", "pure_pursuit")
    if controller_name == "idm":
        return None
    try:
        controller_type = _CONTROLLER_TYPES[controller_name]
    except KeyError as error:
        available = ", ".join(["idm", *_CONTROLLER_TYPES])
        raise ValueError(
            f"Unknown controller '{controller_name}'. Available: {available}"
        ) from error
    controller_params = config.get("controllers", {}).get(controller_name, {})
    return _dataclass_from_config(controller_type, controller_params)


def create_pipeline(
    config: Config,
    controller_name: str | None = None,
) -> PipelineComponents:
    """Create the complete Global Planner -> Local Planner -> Controller chain."""
    return PipelineComponents(
        global_planner=create_global_planner(config),
        local_planner=create_local_planner(config),
        controller=create_controller(config, controller_name),
    )
