# NeuroLoop

:simple-github: [source code](https://github.com/ftsts/neuroloop)

---

## Overview

<img src="/assets/code/ftsts-logo.png"
     alt="NeuroLoop Logo"
     width="48"
     style="float: left; margin-right: 12px;"/>

_Closed-Loop Deep-Brain Stimulation for Controlling Synchronization of Spiking Neurons._

<p align="center">
    <img src="/assets/code/neuroloop/neuroloop-system-overview.png" alt="ftsts-logo" width="300"/>
  <br>
  <em>The overall architecture of this work.</em>
</p>

Implements a closed-loop deep brain stimulation (cl-DBS) system to modulate neural synchronization in a computational model of an Excitatory-Inhibitory network of Leaky Integrate-and-Fire neurons.

Optimizes the open-loop regime described in (_[Schmalz & Jumar, 2019](https://doi.org/10.3389/fncom.2019.00061)_) by implementing a Reinforcement-Learning (RL) driven feedback controller that adjusts stimulation parameters based on real-time measurements of network synchronization.

---

## Results

<div class="grid cards" markdown>

**Training Loss** ![Training Loss Graph](../assets/code/neuroloop/training-loss.png)

**Episode Reward Mean** ![Episode Reward Graph](../assets/code/neuroloop/ep-rew-mean.png)

**Neural Synchronization** ![Sync Graph](../assets/code/neuroloop/kop-1600-400.png)

**Mean Synaptic Weight** ![Syn Wt Graph](../assets/code/neuroloop/w_ie-1600-400.png)

**Neuron Spike Times (pre-stim)** ![pre-stim spikes graph](../assets/code/neuroloop/spikes-pre.png)

**Neuron Spike Times (post-stim)** ![post-stim spikes graph](../assets/code/neuroloop/spikes-post.png)

</div>

Successfully learns to reduce neural synchronization and forces a spiking pattern.

!!! warning

    I have yet to figure out a better way to concisely display results. <br>
    Mermaid XYChart would be preferred if it worked better.

---

## API

---

### Environment

`#!python class gymnasium.Env`  
`#!python class dbsenv.envs.DBSEnv(gymnasium.Env)`  
`#!python class dbsenv.envs.FTSTSEnv(DBSEnv)`

The main Gymnasium class for implementing Reinforcement Learning Agents environments.

The class encapsulates an environment with arbitrary behind-the-scenes dynamics through the `step()` and `reset()` methods.

The main API methods that users of this class need to know are:

- [`step()`](#dbsenvstep) - Updates an environment by taking an action and returning the
  agent's observation, the reward for taking that action, whether the environment
  has terminated or truncated, and auxiliary diagnostic information.
- [`reset()`](#dbsenvreset) - Resets the environment to an initial state, required before the
  first call to [`step()`](#dbsenvstep). Returns an initial observation and auxiliary diagnostic
  information.
- `render()` - Renders the environment to help visualize what the agent sees.
- `close()` - Closes the environment and frees up resources.

---

### Methods

#### DBSEnv.init

```py
DBSEnv(
    sim_config:   SimConfig,
    model_class:  type[NeuralModel],
    model_params: dict | None = None,
    render_mode:  str | None = None
)
```

PARAMETERS:

- `sim_config` – Simulation configuration (timing, sampling, resolution)
- `model_class` – Neural model class implementing the spiking dynamics
- `model_params` – Optional model-specific keyword arguments
- `render_mode` – Optional Gymnasium render mode

#### DBSEnv.reset

```py
DBSEnv.reset(
    seed:    Optional[int] = None,
    options: Optional[dict] = None
) -> tuple[ObsType, dict[str, Any]]:
```

Resets the environment to an initial state (reinitializing plasticity,
stimulation timing, internal counters, etc.), returning an initial observation
and info.

PARAMETERS:

- `seed (optional int)` – The seed that is used to initialize the environment's
  PRNG (_np_random_) and the read-only attribute _np_random_seed_.
- `options (optional dict)` – Additional information to specify how the
  environment is reset.

RETURNS:

- `observation (ObsType)` – Observation of the initial state.
- `info (dict)` – Auxiliary diagnostic information complementing the observation.

#### DBSEnv.step

```py
DBSEnv.step(
    action: ActType
) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]
```

Runs one timestep of the environment's dynamics using the agent's actions.

PARAMETERS:

- `action (ActType)` – an action provided by the agent to update the environment state.

RETURNS:

- `observation (ObsType)` – Population-level neural statistics
- `reward (SupportsFloat)` – The reward as a result of taking the action, encouraging (de)synchronization
- `terminated` – Episode termination flag
- `truncated` – Truncation flag (unused)
- `info` – Diagnostic data (e.g., spike timing)

---

### Attributes

#### DBSEnv.action_space

The actions are continuous and normalized internally before being mapped to
biophysically meaningful stimulation parameters.

| Num | Action                | Unit | Min | Max |
| --- | --------------------- | ---- | --- | --- |
| 0   | Stimulation Amplitude | mV   | 10  | 200 |
| 1   | Stimulation Frequency | Hz   | 5   | 180 |
| 2   | Pulse Width           | μs   | 50  | 500 |

#### DBSEnv.observation_space

Observations are aggregated over a fixed temporal window and normalized
to ensure stable learning dynamics across stimulation regimes.

| Num | Observation                      | Population | Unit     | Min | Max |
| --- | -------------------------------- | ---------- | -------- | --- | --- |
| 0   | Synchrony (Order Parameter)      | Global     | unitless | 0.0 | 1.0 |
| 1   | Mean Excitatory Membrane Voltage | E          | mV       | -80 | 50  |
| 2   | Std. Excitatory Membrane Voltage | E          | mV       | 0.0 | 30  |
| 3   | Mean Inhibitory Membrane Voltage | I          | mV       | -80 | 50  |
| 4   | Std. Inhibitory Membrane Voltage | I          | mV       | 0.0 | 30  |
| 5   | Mean I→E Synaptic Weight         | I→E        | unitless | 0.0 | 1.0 |

---
