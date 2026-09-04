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

"""Reference JAX RL agents you can train on navix environments:
`PPO`, `PQN` (a replay/target-free deep Q-network) and `Dreamer`
(DreamerV3). Each pairs with an `HParams` struct and plugs into
`navix.experiment.Experiment`. `models` holds the shared network pieces,
including the carry-based `Encoder` family (swap `TransformerEncoder` in
for a history-conditioned policy on partially observable tasks).
"""

from .ppo import PPO, PPOHparams as PPOHparams
from .models import (
    Encoder,
    MLPEncoder,
    ConvEncoder,
    TransformerBlock,
    TransformerEncoder,
    ActorCritic,
    QNetwork,
    QMLPEncoder,
    QConvEncoder,
)
from .dreamer import (
    Dreamer,
    DreamerHparams as DreamerHparams,
    WorldModel,
    Actor as DreamerActor,
    Critic as DreamerCritic,
)
from .pqn import PQN, PQNHparams as PQNHparams
