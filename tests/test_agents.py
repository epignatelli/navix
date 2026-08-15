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

from unittest.mock import patch

import jax.numpy as jnp

from navix.agents.agent import Agent, HParams


def test_log_on_train_end_respects_log_frequency():
    # https://github.com/epignatelli/navix/issues/60
    # log_on_train_end used to index into every field of `logs` (a
    # device-to-host transfer per field) for every single recorded step,
    # even though self.log() would immediately discard most of them via
    # the log_frequency check - wandb.log() was correctly only called for
    # the kept steps, but the expensive tree indexing happened regardless.
    # This asserts the *set of steps actually logged* is unchanged by
    # hoisting that check earlier: still exactly the steps where
    # iter/updates % log_frequency == 0.
    n_steps = 10
    log_frequency = 3
    logs = {
        "iter/updates": jnp.arange(n_steps),
        "iter/frames": jnp.arange(n_steps) * 100,
    }
    agent = Agent(hparams=HParams(log_frequency=log_frequency))

    with patch("navix.agents.agent.wandb.log") as mock_log:
        agent.log_on_train_end(logs)

    logged_steps = [int(call.kwargs["step"]) for call in mock_log.call_args_list]
    expected_steps = [s for s in range(n_steps) if s % log_frequency == 0]
    assert logged_steps == expected_steps, (
        f"Expected wandb.log to be called for steps {expected_steps}, "
        f"got {logged_steps}"
    )


if __name__ == "__main__":
    test_log_on_train_end_respects_log_frequency()
