import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
from itertools import permutations

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_torch(np_array):
    return torch.from_numpy(np_array)


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(
            tgt_param.data * (1.0 - tau) + src_param.data * tau
        )


def sample_actions(action, num_samples=10, action_noise=0.1):
    """
    Generate multiple action samples by adding Gaussian noise to the original action.

    Args:
        action: numpy array, original action taken by the agent (action_dim,).
        num_samples: int, number of action samples to generate.
        action_noise: float, standard deviation of Gaussian noise added to actions.

    Returns:
        numpy array: A set of sampled actions (num_samples, action_dim).
    """
    # Convert action to tensor if it's not already
    action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    sampled_actions = []
    for _ in range(num_samples):
        # Sample noisy action
        noisy_action = action + action_noise * torch.randn_like(action)
        noisy_action = noisy_action.clamp(-1, 1)  # Ensure actions are within valid range
        sampled_actions.append(noisy_action.squeeze(0).numpy())  # Remove batch dimension for each sample

    return np.array(sampled_actions)

def build_goal_state_for_spread(obs_batch, n_agents=3, n_landmarks=3):
    """
    From a batch of agent observations [n_agents, obs_dim],
    extract agent positions and landmark positions,
    and construct a goal state where each agent is placed on a landmark.
    
    Returns:
        goal_obs: [n_agents, obs_dim] with updated p_pos and zero velocity.
    """
    obs_i = obs_batch[0] 
    obs_dim = len(obs_i)
    goal_obs = []
    p_vel = np.zeros(2)
    p_pos = obs_i[2:4]
    entity_pos = obs_i[4:4 + 2 * n_landmarks].reshape(n_landmarks, 2)
    landmark_abs = entity_pos + p_pos  # landmark absolute pos

    all_assignments = list(permutations(range(n_landmarks), n_agents))
    centralized_states = []

    for assignment in all_assignments:
        obs_list = []
        for i in range(n_agents):
            agent_obs = obs_batch[i]
            agent_p_pos = agent_obs[2:4]

            assigned_landmark = assignment[i]
            goal_pos = landmark_abs[assigned_landmark]  # assign this landmark

            # Update observation
            new_entity_pos = landmark_abs - goal_pos
            other_pos = []
            for j in range(n_agents):
                if j == i: continue
                other_landmark_pos = landmark_abs[assignment[j]]
                other_pos.append(other_landmark_pos - goal_pos)
            comm = [np.zeros(2) for _ in range(n_agents - 1)]

            obs_i = np.concatenate([
                p_vel, goal_pos,
                new_entity_pos.flatten(),
                np.array(other_pos).flatten(),
                np.array(comm).flatten()
            ])
            obs_list.append(obs_i)

        centralized_state = np.concatenate(obs_list)
        centralized_states.append(centralized_state)

    return centralized_states

def build_goal_state_for_line(obs_batch, n_agents=5, n_landmarks=5, n_obstacles=3):
    """
    Construct a “goal observation” in which each agent is exactly on its
    corresponding landmark and remains stationary.

    Parameters
    ----------
    obs_batch : np.ndarray
        Batch of raw observations with shape (n_agents, obs_dim).
        Layout must match your `observation()` method:
        [p_vel(2), p_pos(2), rel_landmarks(2*n_landmarks),
         rel_obstacles(2*n_obstacles), rel_other_agents(2*(n_agents-1))]
    n_agents : int
        Number of agents in the world.
    n_landmarks : int
        Number of designated goal landmarks (the first few landmarks).
    n_obstacles : int
        Number of static obstacles (corridor walls).

    Returns
    -------
    goal_obs : np.ndarray
        Observations of the same shape but representing the perfect-goal state.
    """

    # -----------------------------------------------------------
    # 1) Recover absolute landmark / obstacle coordinates
    # -----------------------------------------------------------
    # Agent-0’s observation is enough to infer global entity positions because
    # landmarks and walls are static and shared
    obs0 = obs_batch[0]
    p_pos0 = obs0[2:4]  # absolute position of agent-0

    # Relative → absolute for landmarks
    rel_lm = obs0[4 : 4 + 2 * n_landmarks].reshape(n_landmarks, 2)
    landmark_abs = rel_lm + p_pos0  # shape (n_landmarks, 2)

    # Relative → absolute for obstacles
    start_obs = 4 + 2 * n_landmarks
    rel_ob = obs0[start_obs : start_obs + 2 * n_obstacles].reshape(n_obstacles, 2)
    obstacle_abs = rel_ob + p_pos0  # shape (n_obstacles, 2)

    # -----------------------------------------------------------
    # 2) Assign each agent to a landmark: agent-i → landmark-i
    # -----------------------------------------------------------
    goal_positions = landmark_abs[:n_agents]

    # Pre-flatten landmarks / obstacles for faster broadcasting later
    landmark_abs_mat = landmark_abs.reshape(-1, 2)
    obstacle_abs_mat = obstacle_abs.reshape(-1, 2)

    goal_obs_list = []

    # -----------------------------------------------------------
    # 3) Build per-agent goal observation
    # -----------------------------------------------------------
    for i in range(n_agents):
        # Target position and zero velocity for agent-i
        pos_i = goal_positions[i]
        vel_i = np.zeros(2)

        # Recompute relative coordinates **from the goal location**
        rel_lm_i = (landmark_abs_mat - pos_i).reshape(-1)     # landmarks & goals
        rel_ob_i = (obstacle_abs_mat - pos_i).reshape(-1)     # corridor walls

        # Relative positions of the other agents at their own goals
        rel_other_i = []
        for j in range(n_agents):
            if j == i:
                continue
            rel_other_i.append(goal_positions[j] - pos_i)
        rel_other_i = np.concatenate(rel_other_i).reshape(-1)

        # Concatenate in exactly the same order as the environment’s observation
        obs_i_goal = np.concatenate(
            [vel_i, pos_i, rel_lm_i, rel_ob_i, rel_other_i]
        )
        goal_obs_list.append(obs_i_goal)

    # Stack into final (n_agents, obs_dim) array
    goal_obs = np.stack(goal_obs_list, axis=0).reshape(-1)
    return goal_obs


def build_goal_state_for_corridor(obs_batch, n_agents = 3, n_landmarks = 3, n_obstacles = 2):
    """
    Construct a “goal observation” in which each agent is exactly on its
    corresponding landmark and remains stationary.

    Parameters
    ----------
    obs_batch : np.ndarray
        Batch of raw observations with shape (n_agents, obs_dim).
        Layout must match your `observation()` method:
        [p_vel(2), p_pos(2), rel_landmarks(2*n_landmarks),
         rel_obstacles(2*n_obstacles), rel_other_agents(2*(n_agents-1))]
    n_agents : int
        Number of agents in the world.
    n_landmarks : int
        Number of designated goal landmarks (the first few landmarks).
    n_obstacles : int
        Number of static obstacles (corridor walls).

    Returns
    -------
    goal_obs : np.ndarray
        Observations of the same shape but representing the perfect-goal state.
    """

    # -----------------------------------------------------------
    # 1) Recover absolute landmark / obstacle coordinates
    # -----------------------------------------------------------
    # Agent-0’s observation is enough to infer global entity positions because
    # landmarks and walls are static and shared
    obs0 = obs_batch[0]
    p_pos0 = obs0[2:4]  # absolute position of agent-0

    # Relative → absolute for landmarks
    rel_lm = obs0[4 : 4 + 2 * n_landmarks].reshape(n_landmarks, 2)
    landmark_abs = rel_lm + p_pos0  # shape (n_landmarks, 2)

    # Relative → absolute for obstacles
    start_obs = 4 + 2 * n_landmarks
    rel_ob = obs0[start_obs : start_obs + 2 * n_obstacles].reshape(n_obstacles, 2)
    obstacle_abs = rel_ob + p_pos0  # shape (n_obstacles, 2)

    # -----------------------------------------------------------
    # 2) Assign each agent to a landmark: agent-i → landmark-i
    # -----------------------------------------------------------
    goal_positions = landmark_abs[:n_agents]

    # Pre-flatten landmarks / obstacles for faster broadcasting later
    landmark_abs_mat = landmark_abs.reshape(-1, 2)
    obstacle_abs_mat = obstacle_abs.reshape(-1, 2)

    goal_obs_list = []

    # -----------------------------------------------------------
    # 3) Build per-agent goal observation
    # -----------------------------------------------------------
    for i in range(n_agents):
        # Target position and zero velocity for agent-i
        pos_i = goal_positions[i]
        vel_i = np.zeros(2)

        # Recompute relative coordinates **from the goal location**
        rel_lm_i = (landmark_abs_mat - pos_i).reshape(-1)     # landmarks & goals
        rel_ob_i = (obstacle_abs_mat - pos_i).reshape(-1)     # corridor walls

        # Relative positions of the other agents at their own goals
        rel_other_i = []
        for j in range(n_agents):
            if j == i:
                continue
            rel_other_i.append(goal_positions[j] - pos_i)
        rel_other_i = np.concatenate(rel_other_i).reshape(-1)

        # Concatenate in exactly the same order as the environment’s observation
        obs_i_goal = np.concatenate(
            [vel_i, pos_i, rel_lm_i, rel_ob_i, rel_other_i]
        )
        goal_obs_list.append(obs_i_goal)

    # Stack into final (n_agents, obs_dim) array
    goal_obs = np.stack(goal_obs_list, axis=0).reshape(-1)
    return goal_obs


def build_goal_state_for_formation(obs_batch, n_agents = 6, n_landmarks = 6, n_obstacles = 3):
    """
    Construct a “goal observation” in which each agent is exactly on its
    corresponding landmark and remains stationary.

    Parameters
    ----------
    obs_batch : np.ndarray
        Batch of raw observations with shape (n_agents, obs_dim).
        Layout must match your `observation()` method:
        [p_vel(2), p_pos(2), rel_landmarks(2*n_landmarks),
         rel_obstacles(2*n_obstacles), rel_other_agents(2*(n_agents-1))]
    n_agents : int
        Number of agents in the world.
    n_landmarks : int
        Number of designated goal landmarks (the first few landmarks).
    n_obstacles : int
        Number of static obstacles (corridor walls).

    Returns
    -------
    goal_obs : np.ndarray
        Observations of the same shape but representing the perfect-goal state.
    """

    # -----------------------------------------------------------
    # 1) Recover absolute landmark / obstacle coordinates
    # -----------------------------------------------------------
    # Agent-0’s observation is enough to infer global entity positions because
    # landmarks and walls are static and shared
    obs0 = obs_batch[0]
    p_pos0 = obs0[2:4]  # absolute position of agent-0

    # Relative → absolute for landmarks
    rel_lm = obs0[4 : 4 + 2 * n_landmarks].reshape(n_landmarks, 2)
    landmark_abs = rel_lm + p_pos0  # shape (n_landmarks, 2)

    # Relative → absolute for obstacles
    start_obs = 4 + 2 * n_landmarks
    rel_ob = obs0[start_obs : start_obs + 2 * n_obstacles].reshape(n_obstacles, 2)
    obstacle_abs = rel_ob + p_pos0  # shape (n_obstacles, 2)

    # -----------------------------------------------------------
    # 2) Assign each agent to a landmark: agent-i → landmark-i
    # -----------------------------------------------------------
    goal_positions = landmark_abs[:n_agents]

    # Pre-flatten landmarks / obstacles for faster broadcasting later
    landmark_abs_mat = landmark_abs.reshape(-1, 2)
    obstacle_abs_mat = obstacle_abs.reshape(-1, 2)

    goal_obs_list = []

    # -----------------------------------------------------------
    # 3) Build per-agent goal observation
    # -----------------------------------------------------------
    for i in range(n_agents):
        # Target position and zero velocity for agent-i
        pos_i = goal_positions[i]
        vel_i = np.zeros(2)

        # Recompute relative coordinates **from the goal location**
        rel_lm_i = (landmark_abs_mat - pos_i).reshape(-1)     # landmarks & goals
        rel_ob_i = (obstacle_abs_mat - pos_i).reshape(-1)     # corridor walls

        # Relative positions of the other agents at their own goals
        rel_other_i = []
        for j in range(n_agents):
            if j == i:
                continue
            rel_other_i.append(goal_positions[j] - pos_i)
        rel_other_i = np.concatenate(rel_other_i).reshape(-1)

        # Concatenate in exactly the same order as the environment’s observation
        obs_i_goal = np.concatenate(
            [vel_i, pos_i, rel_lm_i, rel_ob_i, rel_other_i]
        )
        goal_obs_list.append(obs_i_goal)

    # Stack into final (n_agents, obs_dim) array
    goal_obs = np.stack(goal_obs_list, axis=0).reshape(-1)
    return goal_obs


def compute_discounted_returns(rewards, dts, gamma=0.99):
    """
    Compute discounted returns for a single episode trajectory.
    :param rewards: [T, n_agents] tensor
    :param dts: [T] tensor of time intervals (dt)
    :param gamma: discount factor (typically gamma = exp(-rho) for continuous time)
    :return: [T, n_agents] tensor of returns
    """
    rewards = torch.stack(rewards).to(device)  # Ensure rewards is a tensor of shape [T, n_agents]
    T, n_agents = rewards.shape[0], 1
    returns = torch.zeros_like(rewards)
    future_return = torch.zeros(n_agents).to(rewards.device)

    for t in reversed(range(T)):
        discount = gamma ** dts[t]  # approximate exp(-rho * dt)
        future_return = rewards[t] * dts[t] + discount * future_return
        returns[t] = future_return

    return returns