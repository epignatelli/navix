<div align="center">
<img width=150px src="https://github.com/epignatelli/navix/assets/26899347/4168c100-f0e6-4bae-9680-2c1a82bba8a4" alt="logo"></img>

# NAVIX: minigrid in JAX
[![CI](https://github.com/epignatelli/navix/actions/workflows/CI.yml/badge.svg)](https://github.com/epignatelli/navix/actions/workflows/CI.yml)
[![CD](https://github.com/epignatelli/navix/actions/workflows/CD.yml/badge.svg)](https://github.com/epignatelli/navix/actions/workflows/CD.yml)
![PyPI version](https://img.shields.io/pypi/v/navix?label=PyPI&color=%230099ab)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?style=flat)](https://arxiv.org/abs/1234.56789) -->

**[Quickstart](#what-is-navix)** | **[Install](#installation)** | **[Examples](#examples)** | **[Docs](https://navix.readthedocs.io)** | **[The JAX ecosystem](#jax-ecosystem-for-rl)** | **[Contribute](#join-us)** | **[Cite](#cite)**

</div>



## What is NAVIX?
NAVIX is a JAX-powered reimplementation of [minigrid](https://github.com/Farama-Foundation/Minigrid). Experiments that took <ins>**1 week**</ins>, now take <ins>**15 minutes**</ins>.   

Key features:
- Performance Boost: NAVIX offers <ins>**over 1000x**</ins> speed increase compared to the original Minigrid implementation, enabling faster experimentation and scaling. You can see a preliminary performance comparison [here](docs/performance.py), and a full benchmarking at [here](benchmarks/).
- XLA Compilation: Leverage the power of XLA to optimize NAVIX computations for many accelerators. NAVIX can run on CPU, GPU, and TPU.
- Autograd Support: Differentiate through environment transitions, opening up new possibilities such as learned world models.

The library is in active development, and we are working on adding more environments and features.
If you want join the development and contribute, please [open a discussion](https://github.com/epignatelli/navix/discussions/new?category=general) and let's have a chat!


## Installation
#### Install JAX
Follow the official installation guide for your OS and preferred accelerator: https://github.com/google/jax#installation.

#### Install NAVIX
```bash
pip install navix
```

Or, for the latest version from source:
```bash
pip install git+https://github.com/epignatelli/navix
```

## Examples
You can view a full set of examples [here](examples/) (more coming), but here are the most common use cases.

### Compiling a collection step
```python
import jax
import navix as nx
import jax.numpy as jnp


def run(seed):
  env = nx.make('MiniGrid-Empty-8x8-v0') # Create the environment
  key = jax.random.PRNGKey(seed)
  timestep = env.reset(key)
  actions = jax.random.randint(key, (N_TIMESTEPS,), 0, env.action_space.n)

  def body_fun(timestep, action):
      timestep = env.step(action)  # Update the environment state
      return timestep, ()

  return jax.lax.scan(body_fun, timestep, actions)[0]

# Compile the entire training run for maximum performance
final_timestep = jax.jit(jax.vmap(run))(jnp.arange(1000))
```

### Compiling a full training run
```python
import jax
import navix as nx
import jax.numpy as jnp
from jax import random

def run_episode(seed, env, policy):
    """Simulates a single episode with a given policy"""
    key = random.PRNGKey(seed)
    timestep = env.reset(key)
    done = False
    total_reward = 0

    while not done:
        action = policy(timestep.observation)
        timestep, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward

def train_policy(policy, num_episodes):
    """Trains a policy over multiple parallel episodes"""
    envs = jax.vmap(nx.make, in_axes=0)(['MiniGrid-MultiRoom-N2-S4-v0'] * num_episodes)
    seeds = random.split(random.PRNGKey(0), num_episodes)

    # Compile the entire training loop with XLA
    compiled_episode = jax.jit(run_episode)
    compiled_train = jax.jit(jax.vmap(compiled_episode, in_axes=(0, 0, None)))

    for _ in range(num_episodes):
        rewards = compiled_train(seeds, envs, policy)
        # ... Update the policy based on rewards ...

# Hypothetical policy function
def policy(observation):
   # ... your policy logic ...
   return action

# Start the training
train_policy(policy, num_episodes=100)
```

### Backpropagation through the environment
```python
import jax
import navix as nx
import jax.numpy as jnp
from jax import grad
from flax import struct


class Model(struct.PyTreeNode):
  @nn.compact
  def __call__(self, x):
    # ... your NN here

model = Model()
env = nx.environments.Room(16, 16, 8)

def loss(params, timestep):
  action = jnp.asarray(0)
  pred_obs = model.apply(timestep.observation)
  timestep = env.step(timestep, action)
  return jnp.square(timestep.observation - pred_obs).mean()

key = jax.random.PRNGKey(0)
timestep = env.reset(key)
params = model.init(key, timestep.observation)

gradients = grad(loss)(params, timestep)
```

## JAX ecosystem for RL
NAVIX is not alone and part of an ecosystem of JAX-powered modules for RL. Check out the following projects:
- Environments:
  - [Gymnax](https://github.com/RobertTLange/gymnax): a broad range of RL environments
  - [Brax](https://github.com/google/brax): a physics engine for robotics experiments
  - [EnvPool](https://github.com/sail-sg/envpool): a set of various batched environments
  - [Craftax](https://github.com/MichaelTMatthews/Craftax): a JAX reimplementation of the game of [Crafter](https://github.com/danijar/crafter)
  - [Jumanji](https://github.com/instadeepai/jumanji): another set of diverse environments
  - [PGX](https://github.com/sotetsuk/pgx): board games commonly used for RL, such as backgammon, chess, shogi, and go
  - [JAX-MARL](https://github.com/FLAIROx/JaxMARL): multi-agent RL environments in JAX
  - [Xland-Minigrid](https://github.com/corl-team/xland-minigrid/): a set of JAX-reimplemented grid-world environments
  - [Minimax](https://github.com/facebookresearch/minimax):  a JAX library for RL autocurricula with 120x faster baselines
- Agents:
  - [PureJaxRl](https://github.com/luchris429/purejaxrl): proposing fullly-jitten training routines
  - [Rejax](https://github.com/keraJLi/rejax): a suite of diverse agents, among which, DDPG, DQN, PPO, SAC, TD3
  - [Stoix](https://github.com/EdanToledo/Stoix): useful implementations of popular single-agent RL algorithms in JAX
  - [JAX-CORL](https://github.com/nissymori/JAX-CORL): lean single-file implementations of offline RL algorithms with solid performance reports
  - [Dopamine](https://github.com/google/dopamine): a research framework for fast prototyping of reinforcement learning algorithms
  

## Join Us!

NAVIX is actively developed. If you'd like to contribute to this open-source project, we welcome your involvement! Start a discussion or open a pull request.

Please, consider starring the project if you like NAVIX!

## Cite
If you use `navix` please cite it as:

```bibtex
@misc{pignatelli2023navix,
  author = {Pignatelli, Eduardo},
  title = {Navix: Scaling gridworld navigation with JAX},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/epignatelli/navix}}
  }
