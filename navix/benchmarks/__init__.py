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

"""A `Benchmark` (`navix.benchmarks.benchmark`) is a preset experimental
setup - a fixed set of environments, a fixed training budget - that
scores an algorithm against it, rather than the single-environment,
single-config runs `Experiment` (`navix/experiment.py`) already supports.
`Benchmark.run` is an orchestration layer *over* `Experiment` (one
`Experiment` per environment, budget-overridden), not a replacement for
it.

This is the "Benchmark" issue #130 (the navix leaderboard proposal)
already scoped conceptually - "a scenario-specific function that takes a
trained agent and returns the metric(s) used to rank it" - implemented
here for the simplest of #130's four protocols, from-scratch training,
as three presets (`Navix1M`, `Navix100K`, `Navix1K`) that differ only in
`budget`. The other protocols (zero-shot, curriculum, open-ended) are
still open per #130 and not attempted here - though `Benchmark._train_on`
is factored out specifically so a future protocol that needs to run
environments as a *sequence* (curriculum: keep training the same agent
across environments; open-ended: an unbounded/evolving task stream)
mostly needs to override that one method or `run` itself, not
scoring/aggregation too.

`Benchmark` itself is a base class, not something instantiated directly:
`name`/`budget` are fixed per preset (each preset sets them as class
attributes), while `entry`/`env_ids`/`seeds` are set per run - so a
preset is used as `Navix1M(entry).run()`, binding the algorithm to score
before running it, rather than `Benchmark.run` taking the entry as a
call-time argument.

Every algorithm a `Benchmark` runs is wrapped in an `AlgorithmEntry`,
carrying the provenance metadata #130's "Structure (decided)" section
requires of a leaderboard row - name, a GitHub-handle-validated author,
and full commit URLs (not bare SHAs, so they're directly traceable/
clickable and self-describing about which repo they belong to - not
optional for an external algorithm's own commit, which has no implied
repo the way navix's own does) for both the navix commit and the
algorithm implementation's own commit the result was produced against -
plus a link to the paper. `AlgorithmEntry` doesn't carry a
`requirements.txt` reference itself: per `benchmarks/README.md`'s
per-entry folder layout, that file lives alongside whatever script
constructs the entry, discoverable by convention rather than a field.
What comes back is a `BenchmarkResult`, shaped identically
regardless of algorithm - exactly `navix.plotting.MANDATORY_METRICS`
(`returns`, `success_rate`, `episode_length`, `fps`, `wall_time`),
aggregated across every environment the benchmark covers - already
matching the "Performance metric(s)"/"Wall-clock training time" columns
#130's leaderboard table spec calls for, rather than needing that
computed separately later. The full per-environment, per-update detail
each of those five fields is reduced from stays available on
`BenchmarkResult.logs`. The ".sh reproduction script" from #130's spec
isn't a field either - it's whatever `benchmarks/<preset>/*.py` script
constructs the entry and calls
`Navix1M(entry).run()`/`Navix100K(entry).run()`.

Per #130's "we never vendor an external algorithm's code" decision,
`AlgorithmEntry.agent_factory` is how a `Benchmark` stays algorithm-
agnostic without navix owning the algorithm's implementation: for navix's
own agents (PPO/Dreamer/PQN) it builds a `navix.agents` instance; for an
external algorithm (e.g. rejax, called as a dependency) it would instead
wrap that library's own training entrypoint behind the same `Environment
-> Agent`-shaped interface `Experiment` expects, with no source code
copied into navix.
"""
from .benchmark import (
    AlgorithmEntry as AlgorithmEntry,
    Benchmark as Benchmark,
    BenchmarkResult as BenchmarkResult,
    Navix1M as Navix1M,
    Navix100K as Navix100K,
    Navix1K as Navix1K,
)
