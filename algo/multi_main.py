import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from continuous_env.make_env import make_env
import argparse, datetime
import numpy as np
import torch, os

# === Imports Updated ===
from algo.vip.agent import vip_agent
from algo.utils import *
from copy import deepcopy

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================
# GPU memory monitoring utility
# ===============================
def print_gpu_memory(memory):
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")
    print(f"Memory Length: {len(memory)/1e4} e4")


# ======================================================
# Compute discounted returns for continuous-time setting
# ======================================================
def compute_discounted_returns(rewards, dts, gamma=0.99):
    """
    Compute discounted returns for a single episode trajectory.

    Args:
        rewards: list of tensors, each with shape [n_agents]
        dts: list of tensors, each with shape [1], representing delta_t
        gamma: discount factor (gamma ≈ exp(-rho) for continuous-time RL)

    Returns:
        returns: tensor with shape [T, n_agents]
    """
    rewards = torch.stack(rewards).to(device)  # [T, n_agents]
    T, n_agents = rewards.shape[0], 1
    returns = torch.zeros_like(rewards)
    future_return = torch.zeros(n_agents).to(rewards.device)

    for t in reversed(range(T)):
        discount = gamma ** dts[t]  # Approximate exp(-rho * dt)
        future_return = rewards[t] * dts[t] + discount * future_return
        returns[t] = future_return

    return returns


# ============================================
# Compute time-to-go sequence for each timestep
# ============================================
def compute_time_to_go_sequence(delta_ts, T):
    """
    Args:
        delta_ts: array-like, shape [num_steps], delta_t at each step
        T: total time horizon

    Returns:
        time_to_go: array-like, shape [num_steps], remaining time at each step
    """
    cumulative_time = np.cumsum(delta_ts)
    time_to_go = T - cumulative_time
    return time_to_go


def main(args):
    # Create environment
    env = make_env(args.scenario, args.seed)

    # Create model directory
    model_dir = f"./trained_model/{args.algo}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Set environment seed
    env.world.seed = args.seed

    # Environment dimensions
    n_agents = env.n
    n_actions = env.world.dim_p
    n_states = env.observation_space[0].shape[0]
    args.n_agents = n_agents

    # ===================
    # Set random seeds
    # ===================
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # =========================
    # Initialize agent
    # =========================
    if args.algo == "vip":
        model = vip_agent(n_states, n_actions, n_agents, args, env)

    print(f"Algorithm initialized: {args.algo}")

    # Time horizon
    T = 5
    model.T = T
    num_steps = args.episode_length
    model.model_dir = model_dir

    # Sample delta_t sequence (fixed across episodes unless resampled)
    delta_ts = np.random.dirichlet(np.ones(num_steps)) * T
    time_to_go = compute_time_to_go_sequence(delta_ts, T)

    episode = 0
    total_step = 0

    while episode < args.max_episodes:
        state = env.reset()

        # Trajectory buffers (on-policy)
        trajectory_obs = []
        trajectory_next_obs = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_dts = []
        trajectory_constraints = []

        episode += 1
        step = 0

        accum_reward = 0
        continued_reward = 0
        uncontinued_reward = 0

        if episode == 3000:
            print("Debug checkpoint reached.")

        while True:
            if args.mode == "train":
                # Select action (with exploration noise if enabled)
                action = model.choose_action(state, delta_ts[step])

                # Environment step with continuous-time dynamics
                next_state, reward, done, con_constraints = env.step_con(
                    deepcopy(action), delta_ts[step]
                )
                con_reward, dis_reward = reward

                step += 1
                total_step += 1

                accum_reward += np.sum(con_reward + dis_reward)
                continued_reward += np.sum(con_reward)
                uncontinued_reward += np.sum(dis_reward)

                # Collect trajectory for on-policy update
                if args.algo in ['vip']:
                    obs = torch.from_numpy(np.stack(state)).float().to(device)
                    obs_ = torch.from_numpy(np.stack(next_state)).float().to(device)

                    rw_tensor = torch.tensor(con_reward).float().to(device)
                    ac_tensor = torch.FloatTensor(action).to(device)
                    delta_t_tensor = torch.FloatTensor([delta_ts[step - 1]]).to(device)
                    constraint_tensor = torch.tensor(con_constraints).to(device)

                    trajectory_obs.append(obs)
                    trajectory_next_obs.append(obs_)
                    trajectory_actions.append(ac_tensor)
                    trajectory_rewards.append(rw_tensor)
                    trajectory_dts.append(delta_t_tensor)
                    trajectory_constraints.append(constraint_tensor)

                else:
                    # Fallback for other algorithms
                    model.memory.push(state, action, reward, next_state, done)

                state = next_state

                # Episode termination condition
                if step >= args.episode_length or (True in done):
                    if args.algo in ['vip']:
                        returns_tensor = compute_discounted_returns(
                            rewards=trajectory_rewards,
                            dts=trajectory_dts,
                            gamma=args.gamma
                        )

                        # Prepare training batch
                        batch = [
                            trajectory_obs,
                            trajectory_actions,
                            trajectory_next_obs,
                            trajectory_rewards,
                            trajectory_dts,
                            returns_tensor
                        ]

                        # Update agent
                        losses = model.update(batch)
                        torch.cuda.empty_cache()

                        print(f"[Episode {episode:05d}] reward {accum_reward:.4f}")

                    # Save model periodically
                    if episode % args.save_interval == 0:
                        model.save_model(episode)

                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--scenario', default="target", type=str)
    parser.add_argument('--max_episodes', default=30000, type=int)
    parser.add_argument('--algo', default="vip", type=str)
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--episode_length', default=50, type=int)
    parser.add_argument('--memory_length', default=int(1e4), type=int)

    parser.add_argument('--tau', default=0.001, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--seed', default=120, type=int)

    parser.add_argument('--a_lr', default=1e-4, type=float)
    parser.add_argument('--c_lr', default=1e-3, type=float)
    parser.add_argument('--lr_dynamics', default=1e-3, type=float)
    parser.add_argument('--lr_reward', default=1e-3, type=float)
    parser.add_argument('--lr_cost', default=1e-3, type=float)

    parser.add_argument('--return_factor', default=15, type=float)
    parser.add_argument('--z_lowerbound', default=0.0, type=float)
    parser.add_argument('--z_bias', default=0.0, type=float)
    parser.add_argument('--z_min', default=0.0, type=float)
    parser.add_argument('--z_max', default=50.0, type=float)

    parser.add_argument('--plot_frequency', default=1000, type=int)
    parser.add_argument('--normal_factor', default=2, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--rnn_hidden_size', default=64, type=int)

    parser.add_argument('--render_flag', default=False, type=bool)
    parser.add_argument('--use_wandb', action="store_true")
    parser.add_argument('--exploration_steps', default=1000, type=int)
    parser.add_argument('--noise_level', default=0.1, type=float)

    parser.add_argument('--ablation_hjb', default=False, type=bool)
    parser.add_argument('--ablation_target', default=False, type=bool)
    parser.add_argument('--ablation_vgi', default=False, type=bool)

    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--epsilon_decay', default=10000, type=int)

    parser.add_argument('--tensorboard', action="store_true")
    parser.add_argument('--ablation', action="store_true")
    parser.add_argument('--relu', action="store_true")

    parser.add_argument('--save_interval', default=3000, type=int)
    parser.add_argument('--model_episode', default=300000, type=int)
    parser.add_argument('--episode_before_train', default=10, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

    args = parser.parse_args()
    main(args)
