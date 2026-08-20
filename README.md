# ad-planner-playground

A modular autonomous-driving planning playground built on
[HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv).

![python_b1vfDhFMab](README.assets/python_b1vfDhFMab.gif)

## Pipeline

The Ego vehicle is organized as a replaceable processing pipeline:

```text
GroundTruthDetector
    -> State Predictor
    -> Global Planner
    -> Local Planner
    -> Trajectory Controller
    -> HighwayEnv ContinuousAction
```

The current implementation includes:

- Ground-truth obstacle detection.
- EKF-based multi-step NPC trajectory prediction.
- A fixed-target-lane global planner.
- Time-parameterized Frenet candidate generation using quintic polynomials.
- Time-indexed collision checking using the full vehicle footprint.
- PID and Pure Pursuit trajectory controllers.
- An IDM/MOBIL fallback mode that bypasses the custom planning pipeline.
- Default and randomized NPC behavior policies.
- Visualization of EKF predictions, the selected trajectory, and all candidate
  trajectories.

## Installation

Create the Conda environment:

```powershell
conda env create -f environment.yml
conda activate highwayenv-py312
```

Alternatively, install the Python requirements directly:

```powershell
pip install -r requirements.txt
```

## Run

Run a headless simulation and save the final frame:

```powershell
python highway_main.py
```

Open the interactive Pygame viewer:

```powershell
python highway_main.py --human
```

In human mode, use the arrow keys to select the target lane:

```text
Up arrow    Move the target lane one lane upward
Down arrow  Move the target lane one lane downward
```

The target lane is clamped to the legal lanes of the current road segment.

## Configuration

Runtime configuration is split into two commented JSONC files:

- [`environment_config.jsonc`](environment_config.jsonc) contains only
  HighwayEnv parameters such as lane count, traffic count, and simulation
  frequencies.
- [`ego_config.jsonc`](ego_config.jsonc) contains the complete Ego pipeline
  configuration: Ego mode, state predictor, global planner, local planner, and
  all controller parameters.

The default command loads both files automatically. Custom files can be passed
independently:

```powershell
python highway_main.py `
    --environment-config my_environment.jsonc `
    --ego-config my_ego.jsonc
```

Controller and state can be selected from the command line. The target speed
and initial target lane are configured only in `ego_config.jsonc`:

```text
local_planner.target_speed
global_planner.target_lane
```

For example:

```powershell
python highway_main.py --human `
    --ego-controller pure_pursuit `
    --draw-all-trajectories
```

## Modes

Available Ego controllers:

```text
idm, pid, pure_pursuit
```

Available explicit planning states:

```text
cruise
```

