from dataclasses import dataclass
from typing import Dict, Tuple
import jax
from flax import struct
from flax.training.train_state import TrainState


@dataclass
class HParams:
    debug: bool = False


class Agent(struct.PyTreeNode):
    hparams: HParams

    def train(self, rng: jax.Array) -> Tuple[TrainState, Dict[str, jax.Array]]:
        raise NotImplementedError

    def log_on_train_end(self, logs: Dict[str, jax.Array]):
        raise NotImplementedError
