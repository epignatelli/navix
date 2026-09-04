# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""NAVIX: MiniGrid-style gridworld navigation, reimplemented in JAX.

Every environment is a pure JAX function, so `reset`/`step` compose with
`jax.jit`, `jax.vmap` and `jax.lax.scan`: a whole batch of environments
advances in one fused device kernel, and an agent's rollout loop can be a
single compiled `scan`. This is what makes NAVIX ~1000x faster end-to-end
than a CPU MiniGrid loop for RL training.

Typical use:

```python
import jax
import navix as nx

env = nx.make("Navix-DoorKey-5x5-v0")          # a registered environment
key = jax.random.PRNGKey(0)
timestep = env.reset(key)                        # -> Timestep
timestep = env.step(timestep, jax.numpy.asarray(2))   # apply action 2
```

Layout:

- `navix.environments` - the [`Environment`][navix.environments.environment.Environment]
  class, the [`Timestep`][navix.environments.environment.Timestep] it
  returns, and `make` / `register_env` / `registry` for the built-in
  environment ids.
- The pluggable pieces an `Environment` is assembled from, each a plain
  function you can swap: `navix.observations`, `navix.actions`,
  `navix.transitions`, `navix.rewards`, `navix.terminations`,
  `navix.events`.
- `navix.states` / `navix.entities` / `navix.components` - the `State`
  data model (the true world state an observation is derived from).
- `navix.grid` - array helpers for grid geometry (cropping, rotation,
  line of sight).
- `navix.rendering` - sprite/tile rendering for RGB observations.
- `navix.spaces` - `Space` descriptors for observations, actions and
  rewards.
- `navix.agents` - reference JAX implementations of PPO, PQN and
  DreamerV3, plus `navix.experiment.Experiment` to run and log them.
- `navix.benchmarks` - experimental protocols that pin an environment,
  a frame budget and a metric, so results are comparable across
  algorithms and against the literature.
"""

from . import (
    actions,
    components,
    entities,
    grid,
    observations,
    rewards,
    environments,
    terminations,
    spaces,
    rendering,
    transitions,
    events,
    agents,
)

from .environments.registry import make, register_env, registry
from .experiment import Experiment
from .environments.environment import Environment, Timestep, StepType
from . import benchmarks
from .benchmarks import Benchmark, BenchmarkResult, TrainingCurve, AlgorithmEntry, Navix1M, Navix100K

# navix.plotting/`from navix import plotting` used to be the old, now-
# removed top-level navix/plotting.py module - restored as an alias to
# its new home, navix.benchmarks.plotting, so existing callers of
# navix.plotting.plot_metric/plot_metrics/plot_dashboard keep working.
# Not a full re-export: the old module's derive_scalar_metrics moved to
# navix.agents.agent.derive_episodic_metrics (renamed, not just
# relocated - it's load-bearing for Experiment.run_hparam_search too,
# not just plotting), so navix.plotting.derive_scalar_metrics
# specifically is still gone, not aliased.
from .benchmarks import plotting