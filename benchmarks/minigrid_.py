# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_procgenpy
from __future__ import annotations
from functools import reduce
import operator
import random
import time
from dataclasses import dataclass

from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
import gymnasium as gym
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
import wandb


@dataclass
class Args:
    exp_name: str = "navix-benchmarks"
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "MiniGrid-DoorKey-8x8-v0"
    """the id of the environment"""
    total_timesteps: int = 10_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 1
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    annotations_path: str = ""
    """the path to the LLM annotations file"""


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x)


class Agent(nn.Module):
    def __init__(self, encoder: nn.Module, envs, hidden_size=64):
        super().__init__()
        self.network = encoder
        self.actor = layer_init(
            nn.Linear(hidden_size, envs.single_action_space.n), std=0.01
        )
        self.critic = layer_init(nn.Linear(hidden_size, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)  # "bhwc" -> "bchw"
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
        )


class FlattenObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(reduce(operator.mul, env.observation_space.shape, 1),),  # type: ignore
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.flatten()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    wandb.init(
        project=args.exp_name,
        group="minigrid",
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Load environments
    envs = gym.vector.make(
        args.env_id,
        max_episode_steps=100,
        num_envs=args.num_envs,
        asynchronous=True,
        wrappers=[FullyObsWrapper, ImgObsWrapper, FlattenObsWrapper],
    )
    envs.is_vector_env = True
    envs = RecordEpisodeStatistics(envs)
    if args.capture_video:
        envs = RecordVideo(envs, f"videos/{run_name}")
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    encoder = MLPEncoder(envs.single_observation_space.shape[0])
    agent = Agent(encoder, envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # TRAINING LOOP

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    action_size: int = envs.single_action_space.n  # type: ignore
    actions = torch.zeros(
        (
            args.num_steps,
            args.num_envs,
        ),
        dtype=torch.int32,
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32).to(
        device
    )
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32).to(
        device
    )
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.bool).to(device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32).to(
        device
    )

    # TRY NOT TO MODIFY: start the game
    frames = 0
    reset_obs, _ = envs.reset()
    next_obs = torch.Tensor(reset_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    training_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        start_time = time.time()
        logs = {}
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            frames += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_terminated, next_truncated, info = envs.step(
                action.cpu().numpy()
            )
            reward = torch.tensor(reward)
            rewards[step] = reward.to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(np.logical_or(next_terminated, next_truncated)).to(
                device
            )

            # for item in info:
            if "episode" in info:
                env_mask = info["_episode"]  # mask for terminal episodes
                returns = info["episode"]["r"][env_mask].mean()
                length = info["episode"]["l"][env_mask].mean()
                is_success = reward[env_mask] >= 1.0
                logs["perf/returns"] = returns
                logs["perf/episode_length"] = length
                logs["perf/success_hits"] = is_success.int().sum()
                logs["perf/success_rate"] = is_success.float().mean()
                print(
                    f"global_step={frames}, episodic_return={returns}", 
                    f"episode_length={length}"
                )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1].float()
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        logs["ppo/total_loss"] = loss.item()
        logs["ppo/value_loss"] = v_loss.item()
        logs["ppo/actor_loss"] = pg_loss.item()
        logs["ppo/entropy"] = entropy_loss.item()
        logs["ppo/approx_kl"] = approx_kl.item()
        logs["ppo/clipfrac"] = np.mean(clipfracs)

        logs["iter/frames"] = frames
        logs["iter/epochs"] = iteration * args.update_epochs
        logs["iter/updates"] = iteration * args.update_epochs * args.num_minibatches
        logs["iter/learing_rate"] = optimizer.param_groups[0]["lr"]

        logs["iter/fps"] = int(b_inds.size / (time.time() - start_time))
        wandb.log(logs)

    envs.close()
    print(f"Training completed in {time.time() - training_time} seconds")
    wandb.finish()
