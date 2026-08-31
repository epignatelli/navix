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