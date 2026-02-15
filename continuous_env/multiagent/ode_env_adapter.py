import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEMultiAgentAdapter:
    def __init__(self, env, policy_fn, reward_fn=None, device="cpu"):
        """
        :param env: original MPE world object (contains agents and their state)
        :param policy_fn: callable function(state) -> action, batched over agents
        :param reward_fn: optional callable(state, action) -> reward
        :param device: torch device
        """
        self.env = env
        self.policy_fn = policy_fn
        self.reward_fn = reward_fn
        self.device = device
        self.n_agents = len(self.env.world.policy_agents)
        self.state_dim = self.env.observation_space[0].shape[0]
        self.action_dim = self.env.world.dim_p

    def get_initial_state_tensor(self):
        """Extract initial states from the world."""
        states = self.env.reset()
        return torch.tensor(states).float().to(self.device)  # shape: [n_agents, state_dim]

    def rollout(self, t_points):
        """
        Roll out the full trajectory over specified time points.
        :param t_points: 1D tensor of time stamps
        :return: s_traj [T, n_agents, state_dim], a_traj [T, n_agents, action_dim], r_traj [T, n_agents]
        """
        s0 = self.get_initial_state_tensor()  # [n_agents, state_dim]
        actions = []

        def ode_func(t, s_flat):
            s = s_flat.view(self.n_agents, self.state_dim)  # [n_agents, state_dim]
            a = self.policy_fn(s)  # [n_agents, action_dim]
            actions.append(a)
            dsdt = []

            for i, agent in enumerate(self.env.world.policy_agents):
                mass = agent.mass if hasattr(agent, "mass") else 1.0
                acc = a[i] / mass
                pos_dot = s[i, self.action_dim:]
                vel_dot = acc
                dsdt.append(torch.cat([pos_dot, vel_dot], dim=-1))

            return torch.stack(dsdt).view(-1)  # flatten for odeint

        s_traj = odeint(ode_func, s0.view(-1), t_points).to(self.device)  # [T, n_agents * state_dim]
        s_traj = s_traj.view(len(t_points), self.n_agents, self.state_dim)

        a_traj = torch.stack(actions).to(self.device)  # [T, n_agents, action_dim]

        if self.reward_fn is not None:
            r_traj = self.reward_fn(s_traj, a_traj)  # [T, n_agents]
        else:
            r_traj = torch.zeros_like(a_traj[..., 0])  # placeholder: [T, n_agents]

        return s_traj, a_traj, r_traj
