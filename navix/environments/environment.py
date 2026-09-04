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
"""The `Environment` class and the `Timestep` it produces.

An `Environment` is a frozen JAX pytree: it holds the grid geometry
(`height`, `width`, `max_steps`) and a set of *pluggable functions* -
`observation_fn`, `reward_fn`, `termination_fn`, `transitions_fn`,
`action_set` - that together define the task. `reset` and `step` are pure
functions of `(key)` / `(timestep, action)`, so they compose with
`jax.jit`, `jax.vmap` (a batch of environments) and `jax.lax.scan` (a
whole rollout in one compiled loop).

Concrete environments (`navix.environments.Empty`, `DoorKey`, ...)
subclass `Environment` and implement `_reset` to lay out their grid;
everything else is inherited. Build one with `navix.make(id)` or
`SomeEnv.create(...)`.
"""

from __future__ import annotations

import abc
from typing import Any, Callable, Dict, Tuple
import jax
import jax.numpy as jnp
from jax import Array
from flax import struct


from .. import rewards, terminations, observations, transitions
from ..rendering.cache import RenderingCache, TILE_SIZE
from ..states import EventsManager, State
from ..actions import DEFAULT_ACTION_SET
from ..spaces import Space, Discrete, Continuous
from ..entities import EntityIds

# Calculate maximum entity ID once at module level for efficiency
# Use vars() to get only class attributes (not inherited ones) and filter for Array instances
MAX_CATEGORICAL_VALUE = 1 + max(
    int(value) for value in vars(EntityIds).values() if isinstance(value, Array)
)


class StepType(struct.PyTreeNode):
    """The three kinds of timestep, stored as `Timestep.step_type`. The
    distinction between the two "episode over" cases matters for
    bootstrapping a value function: bootstrap through a `TRUNCATION`,
    but not through a `TERMINATION`."""

    TRANSITION = jnp.asarray(0)
    """The episode continues; `step` will keep advancing it."""
    TRUNCATION = jnp.asarray(1)
    """The episode was cut off at `max_steps` while still ongoing. The
    final state is *not* absorbing - its value is non-zero - so a value
    estimate should bootstrap through it."""
    TERMINATION = jnp.asarray(2)
    """The episode reached a genuine terminal (absorbing) state - the
    goal, lava, a wrong toggle, etc. Its value is 0; do not bootstrap."""


class Timestep(struct.PyTreeNode):
    """What `Environment.reset` and `Environment.step` return - one moment
    in a trajectory.

    Read it as the *result* of the transition that produced it: `state`
    and `t` describe the world now, and `action` / `reward` / `step_type`
    are the action that got here and its consequences. So after
    `ts_next = env.step(ts, a)`: `ts_next.state` is $s_{t+1}$,
    `ts_next.action` is $a$ (`ts.action` is discarded), `ts_next.reward`
    is $R(s_t, a, s_{t+1})$, and `ts_next.observation` is the observation
    of $s_{t+1}$.

    Every field is a JAX array (or a pytree of them), so a `Timestep`
    `vmap`s over a batch of environments and `scan`s over time.

    Attributes:
        t: steps since the last reset, `i32[]` (or `i32[batch]` when
            vmapped). `0` exactly on a `reset` output.
        observation: the agent's view of `state`, produced by the
            environment's `observation_fn`. Shape and dtype are described
            by `env.observation_space`; for a `pomdp` observation
            function this is a partial (first-person, cropped) view.
        action: the action that led to this timestep, `i32[]`. On a
            `reset` output it is a placeholder `0`.
        reward: the scalar reward for the transition into this timestep,
            `f32[]`, produced by the environment's `reward_fn` (bounds
            given by `env.reward_space`).
        step_type: a `StepType` value (`0`/`1`/`2`) - see `StepType`.
        state: the full `State` (the true MDP state); the observation is
            a function of this. Also carries the PRNG `key` and the
            rendering `cache`.
        info: a plain dict for extra per-step quantities. navix populates
            `info["return"]` (undiscounted return so far this episode);
            you may add your own keys.
    """

    t: Array
    observation: Array
    action: Array
    reward: Array
    step_type: Array
    state: State
    info: Dict[str, Any] = struct.field(default_factory=dict)

    def is_truncation(self) -> Array:
        """`True` where `step_type == StepType.TRUNCATION` (episode cut off
        at `max_steps`). Boolean, same batch shape as `step_type`."""
        return self.step_type == StepType.TRUNCATION

    def is_termination(self) -> Array:
        """`True` where `step_type == StepType.TERMINATION` (genuine
        absorbing terminal state). Boolean, same batch shape as
        `step_type`."""
        return self.step_type == StepType.TERMINATION

    def is_transition(self) -> Array:
        """`True` where the episode is still ongoing
        (`step_type == StepType.TRANSITION`). Boolean."""
        return self.step_type == StepType.TRANSITION

    def is_done(self) -> Array:
        """`True` where the episode has ended for either reason -
        truncation or termination. This is the mask you use to segment
        trajectories or to stop bootstrapping a return."""
        return jnp.logical_or(self.is_truncation(), self.is_termination())

    def is_start(self) -> Array:
        """`True` on the first timestep of an episode (`t == 0`) - both
        the initial `reset` and every post-autoreset step. Robust to
        navix deferring autoreset by one `step` call, unlike checking a
        previous step's `is_done()`."""
        return self.t == 0


class Environment(struct.PyTreeNode):
    """A gridworld task as a frozen JAX pytree.

    The task is defined by five pluggable pieces, each a plain function
    you can override per-`make`/`create` call:

    - `observation_fn(state) -> Array` - what the agent sees.
    - `transitions_fn(state, action, action_set) -> State` - the world
      dynamics (applies `action_set[action]`, then any stochastic
      entity updates).
    - `reward_fn(prev_state, action, state) -> f32[]` - the reward for a
      transition. The `(prev_state, action, state)` triple is the
      convention across `navix.rewards` / `terminations` / `events`:
      `prev_state` is $s_t$, `state` is $s_{t+1}$.
    - `termination_fn(prev_state, action, state) -> bool[]` - whether
      $s_{t+1}$ is a genuine terminal state (truncation at `max_steps`
      is added on top automatically).
    - `action_set` - the tuple of `state -> state` primitives an integer
      action indexes into.

    Subclasses only implement `_reset` (the initial grid layout);
    `reset`/`step` are inherited. Instances are immutable - use
    `env.replace(...)` (from `flax.struct`) to get a modified copy.

    Attributes:
        height: grid height in cells, including the surrounding wall.
        width: grid width in cells, including the surrounding wall.
        max_steps: truncation horizon - `step` returns `StepType.TRUNCATION`
            once `t >= max_steps`. Default (via `create`) is
            `4 * height * width`.
        observation_space: `Space` describing `observation_fn`'s output
            (shape, dtype, bounds).
        action_space: `Discrete` over `len(action_set)`.
        reward_space: `Continuous` bound on the reward, `[-1, 1]` by
            default.
        disable_autoreset: if `False` (default), calling `step` on a
            timestep whose episode already ended returns a fresh `reset`
            instead of stepping. Set `True` to handle episode boundaries
            yourself.
        gamma: discount factor. Not used by `step` itself - carried here
            so agents and `reward_fn`s (e.g. time-discounted goal
            rewards) can read it off the environment.
        penalty_coeff: if non-zero, a terminating reward is reduced by
            `penalty_coeff * (t / max_steps)`, i.e. finishing later is
            worth less. `0.0` disables it.
        observation_fn: `state -> observation`. One of the functions in
            `navix.observations`.
        reward_fn: `(prev_state, action, state) -> f32[]`.
        termination_fn: `(prev_state, action, state) -> bool[]`.
        transitions_fn: `(state, action, action_set) -> state`. Usually
            `transitions.deterministic_transition` or
            `transitions.stochastic_transition` (the default, which also
            moves balls).
        action_set: tuple of `state -> state` callables; an integer
            action `a` applies `action_set[a]`.
    """

    height: int = struct.field(pytree_node=False)
    width: int = struct.field(pytree_node=False)
    max_steps: int = struct.field(pytree_node=False)
    observation_space: Space = struct.field(pytree_node=False)
    action_space: Space = struct.field(pytree_node=False)
    reward_space: Space = struct.field(pytree_node=False)
    disable_autoreset: bool = struct.field(pytree_node=False, default=False)
    gamma: float = struct.field(pytree_node=False, default=0.99)
    penalty_coeff: float = struct.field(pytree_node=False, default=0.0)
    observation_fn: Callable[[State], Array] = struct.field(
        pytree_node=False, default=observations.none
    )
    reward_fn: Callable[[State, Array, State], Array] = struct.field(
        pytree_node=False, default=rewards.DEFAULT_TASK
    )
    termination_fn: Callable[[State, Array, State], Array] = struct.field(
        pytree_node=False, default=terminations.DEFAULT_TERMINATION
    )
    transitions_fn: Callable[
        [State, Array, Tuple[Callable[[State], State], ...]], State
    ] = struct.field(pytree_node=False, default=transitions.DEFAULT_TRANSITION)
    action_set: Tuple[Callable[[State], State], ...] = struct.field(
        pytree_node=False, default=DEFAULT_ACTION_SET
    )

    @classmethod
    def create(
        cls,
        height: int,
        width: int,
        max_steps: int | None = None,
        observation_fn: Callable[[State], Array] = observations.symbolic,
        reward_fn: Callable[[State, Array, State], Array] = rewards.DEFAULT_TASK,
        termination_fn: Callable[
            [State, Array, State], Array
        ] = terminations.DEFAULT_TERMINATION,
        transitions_fn: Callable[
            [State, Array, Tuple[Callable[[State], State], ...]], State
        ] = transitions.DEFAULT_TRANSITION,
        action_set: Tuple[Callable[[State], State], ...] = DEFAULT_ACTION_SET,
        observation_space: Space | None = None,
        action_space: Space | None = None,
        reward_space: Space | None = None,
        disable_autoreset: bool = False,
        **kwargs,
    ) -> Environment:
        """Builds an environment, filling in the spaces and `max_steps`
        that weren't given.

        This is the shared constructor concrete environments call from
        their own `create`; `navix.make(id, **kwargs)` ends up here too.

        Args:
            height (int): grid height in cells (walls included).
            width (int): grid width in cells (walls included).
            max_steps (int | None): truncation horizon. `None` ->
                `4 * height * width`.
            observation_fn (Callable): `state -> observation`, from
                `navix.observations`. Default `observations.symbolic`.
            reward_fn (Callable): `(prev_state, action, state) -> f32[]`.
            termination_fn (Callable): `(prev_state, action, state) -> bool[]`.
            transitions_fn (Callable): `(state, action, action_set) -> state`.
            action_set (tuple[Callable, ...]): `state -> state`
                primitives indexed by the integer action.
            observation_space (Space | None): `None` infers it from
                `observation_fn` and the grid size (works for the
                built-in observation functions; pass one explicitly for
                a custom `observation_fn`).
            action_space (Space | None): `None` -> `Discrete(len(action_set))`.
            reward_space (Space | None): `None` -> `Continuous((), -1, 1)`.
            disable_autoreset (bool): see the class attribute.
            **kwargs: extra fields forwarded to the subclass constructor
                (e.g. `gamma`, `penalty_coeff`, or an environment's own
                layout options like `random_start`).

        Returns:
            Environment: the constructed environment.
        """
        if observation_space is None:
            observation_space = cls._get_obs_space_from_fn(
                width, height, observation_fn
            )
        if action_space is None:
            action_space = Discrete.create(len(action_set))
        if reward_space is None:
            reward_space = Continuous.create(
                shape=(), minimum=jnp.asarray(-1.0), maximum=jnp.asarray(1.0)
            )
        if max_steps is None:
            max_steps = 4 * height * width
        return cls(
            height=height,
            width=width,
            max_steps=max_steps,
            observation_fn=observation_fn,
            reward_fn=reward_fn,
            termination_fn=termination_fn,
            transitions_fn=transitions_fn,
            action_set=action_set,
            observation_space=observation_space,
            action_space=action_space,
            reward_space=reward_space,
            disable_autoreset=disable_autoreset,
            **kwargs,
        )

    @abc.abstractmethod
    def _reset(self, key: Array, cache: RenderingCache | None = None) -> Timestep:
        """Lays out the initial grid and returns the first `Timestep`.
        Implemented by each concrete environment; call `reset` (which
        wraps this) rather than this directly.

        Args:
            key (Array): PRNG key for any randomised placement.
            cache (RenderingCache | None): reuse a rendering cache built
                for a same-shaped grid; `None` builds a fresh one.

        Returns:
            Timestep: `t = 0`, `step_type = TRANSITION`, placeholder
            `action`/`reward`."""
        raise NotImplementedError()

    def reset(self, key: Array, cache: RenderingCache | None = None) -> Timestep:
        """Starts a new episode.

        Args:
            key (Array): a `jax.random` PRNG key. Split it yourself
                across a batch (`jax.vmap(env.reset)(keys)`).
            cache (RenderingCache | None): optional pre-built rendering
                cache (see `_reset`).

        Returns:
            Timestep: the first timestep - `t = 0`, `is_start()` true,
            `info["return"] = 0.0`. `state.key` is a fresh key derived
            from `key`, so the environment's own stochasticity is
            reproducible from the single seed you pass here.
        """
        k1, k2 = jax.random.split(key)
        timestep = self._reset(k1, cache)
        timestep.info["return"] = jnp.asarray(0.0)
        return timestep.replace(state=timestep.state.replace(key=k2))

    def step(self, timestep: Timestep, action: Array) -> Timestep:
        """Advances one timestep, or auto-resets at an episode boundary.

        If `timestep` already ended an episode (its `step_type > 0`) and
        `disable_autoreset` is `False`, this ignores `action` and returns
        a fresh `reset` seeded from `timestep.state.key`. Otherwise it
        applies `action` via `transitions_fn`, then evaluates
        `reward_fn`, `termination_fn` and `observation_fn` on the result.

        Autoreset is deferred by one call: the terminal timestep is
        returned as-is (so you can read its final reward), and the reset
        happens on the *next* `step`. Detect a fresh episode with
        `timestep.is_start()`, not by looking back at `is_done()`.

        Args:
            timestep (Timestep): the current timestep (from `reset` or a
                previous `step`).
            action (Array): an integer action, `i32[]`, in
                `[0, len(action_set))`. Indexes `action_set`.

        Returns:
            Timestep: the next timestep. `info["return"]` accumulates the
            undiscounted episodic reward.
        """
        # autoreset if necessary: 0 = transition, 1 = truncation, 2 = termination
        should_reset = timestep.step_type > 0
        return jax.lax.cond(
            jnp.logical_and(should_reset, jnp.logical_not(self.disable_autoreset)),
            lambda timestep: self.reset(timestep.state.key, timestep.state.cache),
            lambda timestep: self._step(timestep, action),
            timestep,
        )

    def _step(self, timestep: Timestep, action: Array) -> Timestep:
        """The non-autoreset half of `step`: apply `action` to
        `timestep.state` and build the resulting `Timestep` at `t + 1`.
        `step` calls this; call `step`, not this.

        Args:
            timestep (Timestep): the timestep at time $t$.
            action (Array): the integer action $a_t$.

        Returns:
            Timestep: the timestep at time $t + 1$.
        """
        # events are a per-step record - EventsManager's own docstring
        # says "which events happened this timestep" - but
        # EventsManager.merge_event only ever ORs new hits onto
        # whatever a slot already holds (by design, see issue #139:
        # that's what lets two events in the *same* step's transition
        # pipeline both survive), so nothing actually clears a slot
        # back to False between steps unless something does it here.
        # Reset before the transition runs, not after, so a hit
        # recorded during this step's own transition_fn call still
        # counts - only carried-over history from earlier steps is
        # dropped. Without this, a non-terminating reward like
        # rewards.wall_hit_cost would keep firing every subsequent step
        # after the first wall hit, not just the step it happened
        # (terminating conditions like on_goal_reached/on_lava_fall/
        # on_ball_hit were never actually affected by this in practice -
        # the episode ends the first time they fire, so there's no
        # "subsequent step" for the stale True to leak into before
        # autoreset gives every entity a fresh EventsManager anyway).
        reset_state = timestep.state.replace(events=EventsManager())
        # update agents
        state = self.transitions_fn(reset_state, action, self.action_set)
        t = timestep.t + 1

        # calculate termination
        step_type = self.termination(timestep.state, action, state, timestep.t + 1)

        # calculate reward
        reward = self.reward_fn(timestep.state, action, state)
        reward = jax.lax.cond(
            step_type == StepType.TERMINATION,
            lambda reward: reward - self.penalty_coeff * (t / self.max_steps),
            lambda reward: reward,
            reward,
        )

        new_timestep = Timestep(
            t=t,
            state=state,
            action=jnp.asarray(action),
            reward=reward,
            step_type=step_type,
            observation=self.observation_fn(state),
        )

        new_timestep.info["return"] = (
            timestep.info.get("return", jnp.asarray(0.0)) + reward
        )

        # build timestep
        return new_timestep

    def termination(
        self, prev_state: State, action: Array, state: State, t: Array
    ) -> Array:
        """Combines the task's `termination_fn` with the `max_steps`
        truncation into a single `StepType`.

        Args:
            prev_state (State): $s_t$.
            action (Array): $a_t$.
            state (State): $s_{t+1}$.
            t (Array): the step count of `state` (`i32[]`).

        Returns:
            Array: a `StepType` value - `TERMINATION` if `termination_fn`
            fired, else `TRUNCATION` if `t >= max_steps`, else
            `TRANSITION`. Termination takes precedence over truncation.
        """
        terminated = self.termination_fn(prev_state, action, state)
        truncated = t >= self.max_steps
        return terminations.check_truncation(terminated, truncated)

    @staticmethod
    def _get_obs_space_from_fn(
        width: int, height: int, observation_fn: Callable[[State], Array]
    ) -> Space:
        """Infers the `observation_space` for a built-in `observation_fn`
        and grid size. Raises `NotImplementedError` for an unrecognised
        function - pass `observation_space` explicitly to `create` in
        that case."""
        if observation_fn == observations.none:
            return Continuous.create(
                shape=(), minimum=jnp.asarray(0.0), maximum=jnp.asarray(0.0)
            )
        elif observation_fn == observations.categorical:
            return Discrete.create(
                n_elements=MAX_CATEGORICAL_VALUE, shape=(height, width)
            )
        elif observation_fn == observations.categorical_first_person:
            radius = observations.RADIUS
            return Discrete.create(
                n_elements=MAX_CATEGORICAL_VALUE,
                shape=(radius * 2 + 1, radius * 2 + 1),
            )
        elif observation_fn == observations.rgb:
            return Discrete.create(
                256,
                shape=(height * TILE_SIZE, width * TILE_SIZE, 3),
                dtype=jnp.uint8,
            )
        elif observation_fn == observations.rgb_first_person:
            radius = observations.RADIUS
            return Discrete.create(
                n_elements=256,
                shape=((radius * 2 + 1) * TILE_SIZE, (radius * 2 + 1) * TILE_SIZE, 3),
                dtype=jnp.uint8,
            )
        elif observation_fn == observations.symbolic:
            return Discrete.create(
                n_elements=MAX_CATEGORICAL_VALUE,
                shape=(height, width, 3),
                dtype=jnp.uint8,
            )
        elif observation_fn == observations.symbolic_first_person:
            radius = observations.RADIUS
            return Discrete.create(
                n_elements=256,
                shape=(radius * 2 + 1, radius * 2 + 1, 3),
                dtype=jnp.uint8,
            )
        else:
            raise NotImplementedError(
                "Unknown observation space for observation function {}".format(
                    observation_fn
                )
            )
