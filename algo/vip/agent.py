# from algo.maddpg.network import Critic, Actor
from algo.vip.network import RewardNet, DynamicsNet, ValueNet, PolicyNet

import torch
from copy import deepcopy
from torch.optim import Adam
from algo.memory import continuous_ReplayMemory
import os
import torch.nn as nn
import numpy as np
from algo.utils import device
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from algo.utils import *

scale_reward = 0.01


def f_wrapper(x, u, dt, net):
    """Return dx/dt prediction so the Jacobian is taken w.r.t. the derivative."""
    return (net(x, u, dt) - x) / dt  # [B, ND]


# from functorch import jacrev  # torch >=1.13, or use functorch
from torch import vmap
from torch.func import jacrev


def batched_jacobian(f, x, u, dt):
    """
    Compute a batched Jacobian for f w.r.t. x using vmap + jacrev.

    Args:
        f: function handle
        x: [B, D_in]
        u: [B, ...]
        dt: [B, 1] or [B]
    Returns:
        J: [B, D_out, D_in]
    """
    def f_wrapped(xi, ui, dti):
        return f(xi.unsqueeze(0), ui.unsqueeze(0), dti.unsqueeze(0)).squeeze(0)
    return vmap(jacrev(f_wrapped))(x, u, dt)  # [B, D_out, D_in]


def compute_time_to_go_sequence(delta_ts, T):
    """
    Args:
        delta_ts: array-like, [num_steps], delta_t at each step
        T: total time horizon
    Returns:
        time_to_go: array-like, [num_steps], remaining time at each step
    """
    cumulative_time = np.cumsum(delta_ts)  # [dt0, dt0+dt1, dt0+dt1+dt2, ...]
    time_to_go = T - cumulative_time       # remaining time
    return time_to_go


class vip_agent:
    def __init__(self, dim_obs, dim_act, n_agents, args, env):
        self.args = args
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.batch_size = args.batch_size
        self.exploration_steps = args.exploration_steps
        self.episodes_before_train = args.episode_before_train
        self.use_cuda = torch.cuda.is_available()

        # Exploration noise schedule (sigma annealing)
        self.exploration_noise_std = 0.99
        self.sigma_init = 0.5
        self.sigma_min = args.noise_level
        self.sigma_decay_steps = 5000
        self._sigma_scale = self.sigma_init

        # Initialize networks (one set per agent)
        self.cost_nets = [ValueNet(dim_obs, n_agents, self.args.relu) for _ in range(n_agents)]
        self.policy_nets = [PolicyNet(dim_obs, dim_act) for _ in range(n_agents)]
        self.value_nets = [ValueNet(dim_obs, n_agents, self.args.relu) for _ in range(n_agents)]
        self.dynamics_nets = [DynamicsNet(dim_obs, dim_act, n_agents) for _ in range(n_agents)]
        self.reward_nets = [RewardNet(dim_obs, dim_act, n_agents) for _ in range(n_agents)]

        # NOTE: naming kept as-is; single_cost_net currently uses RewardNet
        self.single_cost_net = [RewardNet(dim_obs, dim_act, n_agents) for _ in range(n_agents)]

        # Covariance used for Gaussian exploration in action sampling
        self.cov_matrix = torch.eye(self.n_actions) * self.sigma_min

        # Optimizers
        self.policy_optimizers = [Adam(net.parameters(), lr=args.a_lr) for net in self.policy_nets]
        self.value_optimizers = [Adam(net.parameters(), lr=args.c_lr) for net in self.value_nets]
        self.cost_optimizers = [Adam(net.parameters(), lr=args.lr_cost) for net in self.cost_nets]
        self.dynamics_optimizers = [Adam(net.parameters(), lr=args.lr_dynamics) for net in self.dynamics_nets]
        self.reward_optimizers = [Adam(net.parameters(), lr=args.lr_reward) for net in self.reward_nets]
        self.single_cost_optimizers = [Adam(net.parameters(), lr=args.c_lr) for net in self.single_cost_net]

        self.env = env

        # Epigraph-related bounds (kept as provided)
        self.z_min = args.z_min
        self.z_max = args.z_max

        # Move to GPU if available
        if self.use_cuda:
            for i in range(n_agents):
                self.policy_nets[i].cuda()
                self.value_nets[i].cuda()
                self.reward_nets[i].cuda()
                self.cost_nets[i].cuda()
                self.single_cost_net[i].cuda()
                self.dynamics_nets[i].cuda()
            self.cov_matrix = self.cov_matrix.cuda()

        # Target networks (deep copy)
        self.target_value_nets = deepcopy(self.value_nets)

        # Replay buffer (currently created but not used in the on-policy path shown)
        self.memory = continuous_ReplayMemory(args.memory_length)

        # Misc bookkeeping
        self.var = [1.0 for _ in range(n_agents)]
        self.steps_done = 0
        self.episode_done = 0

    def _decay_sigma(self):
        """Linearly anneal exploration sigma from sigma_init to sigma_min."""
        frac = min(1.0, float(self.episode_done) / float(self.sigma_decay_steps))
        self._sigma_scale = self.sigma_init + frac * (self.sigma_min - self.sigma_init)

    def soft_update_target_value_net(self, tau=0.01):
        """Soft update target value networks: θ' ← τ θ + (1-τ) θ'."""
        for target_param, param in zip(self.target_value_nets.parameters(), self.value_nets.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def load_model(self, episode: int):
        """Load per-agent policy nets and shared nets from state_dict files (legacy loader)."""
        load_episode = episode
        load_dir = os.path.join("trained_model", str(self.args.trained_scenario), str(self.args.algo))
        model_path = load_dir

        # Expected file paths
        policy_paths = [
            f"{model_path}/policy_net[{idx}]_{self.args.load_seed}_{load_episode}.pth"
            for idx in range(self.n_agents)
        ]
        value_paths = f"{model_path}/value_nets_{self.args.load_seed}_{load_episode}.pth"
        dyn_path = f"{model_path}/dynamics_nets_{self.args.load_seed}_{load_episode}.pth"
        rew_path = f"{model_path}/reward_nets_{self.args.load_seed}_{load_episode}.pth"

        path_flag = all(os.path.exists(p) for p in policy_paths) and \
                    os.path.exists(value_paths) and \
                    os.path.exists(dyn_path) and os.path.exists(rew_path)

        if not path_flag:
            print("Model files not found, skipping load.")
            return

        print(f"Loading model (state_dict) from episode {load_episode}...")

        # Use map_location to avoid crash when loading across different devices
        map_loc = getattr(self, "device", None)
        if map_loc is None:
            map_loc = "cpu"

        for idx in range(self.n_agents):
            policy_sd = torch.load(policy_paths[idx], map_location=map_loc)
            self.policy_net[idx].load_state_dict(policy_sd)

        value_sd = torch.load(value_paths, map_location=map_loc)
        dyn_sd = torch.load(dyn_path, map_location=map_loc)
        rew_sd = torch.load(rew_path, map_location=map_loc)

        self.dynamics_nets.load_state_dict(dyn_sd)
        self.reward_nets.load_state_dict(rew_sd)
        self.value_nets.load_state_dict(value_sd)

        print("Model loaded successfully.")

    def load_model2(self, episode):
        """Load a unified checkpoint that stores lists of per-agent state_dicts."""
        load_dir = os.path.join("trained_model", str(self.args.trained_scenario), str(self.args.algo))
        ckpt_path = os.path.join(load_dir, f"ckpt_{self.args.load_seed}_ep{episode}.pt")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        def _load_list(nets, key, strict=True):
            if key not in ckpt:
                if strict:
                    raise KeyError(f"Missing key '{key}' in checkpoint: {ckpt_path}")
                else:
                    print(f"[Warning] Missing key '{key}', skip.")
                    return
            sd_list = ckpt[key]
            if len(sd_list) != len(nets):
                raise RuntimeError(
                    f"[Load Error] '{key}' length mismatch: ckpt={len(sd_list)} "
                    f"current={len(nets)} (n_agents={self.n_agents})"
                )
            for i, net in enumerate(nets):
                net.load_state_dict(sd_list[i], strict=strict)

        _load_list(self.policy_nets, "policy")
        _load_list(self.value_nets, "value")
        _load_list(self.dynamics_nets, "dynamics")
        _load_list(self.reward_nets, "reward")
        _load_list(self.cost_nets, "cost")
        _load_list(self.single_cost_net, "single_cost")

        # Target nets are optional
        _load_list(self.target_value_nets, "target_value", strict=False)

        # Restore misc state (optional)
        if "episode_done" in ckpt:
            self.episode_done = int(ckpt["episode_done"])
        if "_sigma_scale" in ckpt:
            self._sigma_scale = float(ckpt["_sigma_scale"])

        print(f"[Info] Model loaded successfully from: {ckpt_path}")
        return ckpt_path

    def save_model(self, episode):
        """Save a unified checkpoint that stores lists of per-agent state_dicts."""
        save_dir = os.path.join("trained_model", str(self.args.scenario), str(self.args.algo))
        os.makedirs(save_dir, exist_ok=True)

        payload = {
            "meta": {
                "episode": int(episode) if str(episode).isdigit() else str(episode),
                "seed": int(getattr(self.args, "seed", -1)),
                "n_agents": self.n_agents,
                "obs_dim": self.n_states,
                "act_dim": self.n_actions,
            },
            # Per-agent state_dict lists
            "policy":       [net.state_dict() for net in self.policy_nets],
            "value":        [net.state_dict() for net in self.value_nets],
            "target_value": [net.state_dict() for net in self.target_value_nets],  # optional
            "dynamics":     [net.state_dict() for net in self.dynamics_nets],
            "reward":       [net.state_dict() for net in self.reward_nets],
            "cost":         [net.state_dict() for net in self.cost_nets],
            "single_cost":  [net.state_dict() for net in self.single_cost_net],
            # Misc (optional)
            "episode_done": int(getattr(self, "episode_done", 0)),
            "_sigma_scale": float(getattr(self, "_sigma_scale", 0.0)),
        }

        ckpt_path = os.path.join(save_dir, f"ckpt_{self.args.seed}_ep{episode}.pt")
        torch.save(payload, ckpt_path)
        print(f"[Info] Saved checkpoint to: {ckpt_path}")
        return ckpt_path

    def update(self, batch):
        """Run one full update cycle over dynamics, reward, value, VGI, and policy."""
        dynamics_loss = self.dynamics_training(batch)
        reward_loss = self.reward_training(batch)

        value_loss = self.value_training_epigraph(batch)
        tilde_value_loss = self.tilde_value_training(batch)
        vgi_loss = self.target_vgi_training(batch)
        policy_loss = self.policy_training(batch)

        self._decay_sigma()
        self.episode_done += 1

        return {
            "dynamics_loss": dynamics_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "tilde_value_loss": tilde_value_loss,
            "vgi_loss": vgi_loss,
            "policy_loss": policy_loss
        }

    def choose_action(self, state, delta_ts):
        """
        Args:
            state: list/array-like, shape [n_agents, state_dim]
            delta_ts: scalar delta_t shared across agents for the current step
        """
        obs = torch.from_numpy(np.stack(state)).float().to(device)
        obs = obs.view(self.n_agents, -1)  # [n_agents, state_dim]

        delta_ts = torch.from_numpy(np.array(delta_ts)).float().to(device)  # [1] or scalar
        delta_ts = delta_ts.expand(self.n_agents, 1)  # [n_agents, 1]

        # Concatenate obs and dt as policy input extension
        extended_obs = torch.cat([obs, delta_ts], dim=-1)  # [n_agents, state_dim+1]

        actions = torch.zeros(self.n_agents, self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        for i in range(self.n_agents):
            sb = extended_obs[i].detach()

            # Policy outputs mean action
            mean = self.policy_nets[i](sb).squeeze(0)  # [action_dim]

            # Gaussian exploration: mean + covariance
            dist = MultivariateNormal(mean.view(-1), covariance_matrix=self.cov_matrix)

            # Evaluation uses deterministic mean; training samples via rsample (pathwise)
            if self.args.mode == 'eval':
                act = mean
            else:
                act = dist.rsample()

            act = torch.clamp(act, -1.0, 1.0)
            actions[i, :] = act

        self.steps_done += 1
        return actions.data.cpu().numpy()

    def dynamics_training(self, batch):
        """Train dynamics networks to predict x_{t+1} given (x_t, u_t, dt)."""
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        states, actions, next_states, _, dt, _ = batch
        states = torch.stack(states).type(FloatTensor)        # [B, N, D]
        next_states = torch.stack(next_states).type(FloatTensor)  # [B, N, D]
        actions = torch.stack(actions).type(FloatTensor)      # [B, N, A]
        dt = torch.stack(dt).type(FloatTensor)                # [B, 1]

        B, N, D = states.shape
        for i in range(self.n_agents):
            x_t = states.view(B, -1)        # [B, N*D]
            u_t = actions.view(B, -1)       # [B, N*A]
            x_tp1 = next_states.view(B, -1) # [B, N*D]

            x_next_pred = self.dynamics_nets[i](x_t, u_t, dt)  # [B, N*D]

            loss = nn.MSELoss()(x_next_pred, x_tp1)
            self.dynamics_optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dynamics_nets[i].parameters(), 1.0)
            self.dynamics_optimizers[i].step()

        return loss.item()

    def value_training_epigraph(self, batch):
        """
        Supervised value fitting using Monte-Carlo-style integrated returns (no bootstrap here).
        The code uses `returns` passed from outside; it flips sign for monitoring.

        NOTE:
          If `returns` already represent integrated objective, do not multiply by dt again.
        """
        Float = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dts, returns = batch

        B = len(states)
        x_t = torch.stack(states).type(Float).view(B, -1)

        # Flip sign (seems used to convert reward to cost convention)
        int_returns = -returns  # monitoring / convention

        for i in range(self.n_agents):
            per_return = int_returns[:, i].view(B, -1)  # [B,1]
            V_now = self.value_nets[i](x_t)
            loss = F.mse_loss(V_now, per_return)

            self.value_optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_nets[i].parameters(), 1.0)
            self.value_optimizers[i].step()

        return loss.item()

    def tilde_value_training(self, batch):
        """Train value net with an HJB residual-style loss (continuous-time consistency)."""
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dts, returns = batch

        B = len(states)
        gamma = self.args.gamma
        log_gamma = np.log(gamma)

        # l_c is interpreted as instantaneous cost (rate); here set from -rewards
        l_c = -torch.stack(rewards).type(FloatTensor)
        returns = -returns

        for i in range(self.n_agents):
            # Step 1: build tensors and enable gradients for x_t
            x_t = torch.stack(states).type(FloatTensor).view(B, -1).requires_grad_(True)
            dtss = torch.stack(dts).type(FloatTensor).view(B, 1)
            x_tp1 = torch.stack(next_states).type(FloatTensor).view(B, -1)

            # Step 2: compute V and epigraph surrogate (here: V_ret only)
            V_ret = self.value_nets[i](x_t)
            epigraph_pred = V_ret

            # Compute ∇_x V
            S = V_ret.sum()
            grad_x = torch.autograd.grad(
                outputs=S,
                inputs=x_t,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]

            # pz term (kept as constant here)
            grad_z = -1  # scalar (no grad)

            # Handle None gradients
            if grad_x is None:
                grad_x = torch.zeros_like(x_t)

            # Finite-difference dynamics estimate f(x) ≈ (x_{t+1} - x_t)/dt
            f_xt = (x_tp1 - x_t) / dtss

            # HJB residual terms
            H_term = (grad_x * f_xt).sum(dim=1, keepdim=True)
            z_term = grad_z * l_c
            gamma_term = log_gamma * epigraph_pred

            residual = H_term - z_term + gamma_term
            loss_HJB = (residual ** 2).mean()

            # Optimize value net parameters
            self.value_optimizers[i].zero_grad()
            loss_HJB.backward()
            torch.nn.utils.clip_grad_norm_(self.value_nets[i].parameters(), 1.0)
            self.value_optimizers[i].step()

        return loss_HJB.item()

    def target_vgi_training(self, batch):
        """Train value gradients via a Value Gradient Iteration (VGI) style objective."""
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dts, returns = batch

        B = len(states)
        log_gamma = np.log(self.args.gamma)

        for i in range(self.n_agents):
            # Step 1: prepare x, u, dt and enable gradients
            x_t = torch.stack(states).type(FloatTensor).view(B, -1).requires_grad_(True)
            u_t = torch.stack(actions).type(FloatTensor).view(B, -1)
            dtss = torch.stack(dts).type(FloatTensor).view(B, 1)
            x_tp1 = torch.stack(next_states).type(FloatTensor).view(B, -1)

            # Step 2: compute current value gradient ∇_x V(x_t)
            v_ret = self.value_nets[i](x_t)
            grad_v_x = torch.autograd.grad(
                v_ret.sum(), x_t, create_graph=True, retain_graph=True
            )[0]  # [B, N*D]

            # Step 3: compute ∇_x r_hat(x_t,u_t) (here reward net predicts (-cost))
            r_hat = self.reward_nets[i](x_t, u_t, dtss).view(B, 1)
            grad_r_x = torch.autograd.grad(r_hat.sum(), x_t, create_graph=True)[0]

            # Branch mask for combining gradients (kept as all-ones here)
            mask_val = torch.ones_like(v_ret)  # [B,1]
            mask_val_exp = mask_val.expand_as(grad_r_x)
            grad_rc_x = mask_val_exp * grad_r_x

            # Convert to cost-gradient convention (kept as provided)
            grad_rc_x = -grad_rc_x.view(B, -1)

            # Step 4: compute ∇_x V(x_{t+1}) for target gradient propagation
            x_tp1_d = x_tp1.clone().detach().requires_grad_(True)
            v_ret_n = self.value_nets[i](x_tp1_d)
            grad_v_x_n = torch.autograd.grad(v_ret_n.sum(), x_tp1_d, create_graph=True)[0]

            # Step 5: compute Jacobian-vector product J_f(x_t,u_t)^T * ∇V(x_{t+1})
            x_t_dyn = x_t.clone().detach().requires_grad_(True)
            x_t_next = self.dynamics_nets[i](x_t_dyn, u_t, dtss)
            f_pred = (x_t_next - x_t_dyn) / dtss

            Jt_v = torch.autograd.grad(
                outputs=f_pred,
                inputs=x_t_dyn,
                grad_outputs=grad_v_x_n,
                retain_graph=True,
                create_graph=True
            )[0]  # [B, N*D]

            # Continuous-time discounting term gamma^{dt}
            gamma_dt = torch.exp(log_gamma * dtss)

            # VGI target gradient: ∇r * dt + gamma^{dt} * J_f^T ∇V(next)
            g_hat_vec = grad_rc_x * dtss + gamma_dt * Jt_v

            # VGI loss: match current ∇V to target gradient
            vgi_loss = ((grad_v_x - g_hat_vec).pow(2)).mean()

            self.value_optimizers[i].zero_grad()
            vgi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_nets[i].parameters(), 1.0)
            self.value_optimizers[i].step()

        return vgi_loss.item()

    def reward_training(self, batch):
        """Train reward nets to predict instantaneous cost (here: -reward)."""
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dt, returns = batch

        states = torch.stack(states).type(FloatTensor)    # [B, N, D]
        actions = torch.stack(actions).type(FloatTensor)  # [B, N, A]
        costs = -torch.stack(rewards).type(FloatTensor)   # [B, N]
        dt = torch.stack(dt).type(FloatTensor)            # [B, 1]

        B, N, D = states.shape
        for i in range(self.n_agents):
            x_t = states.view(B, -1)   # [B, N*D]
            u_t = actions.view(B, -1)  # [B, N*A]
            cost = costs[:, i]

            r_hat = self.reward_nets[i](x_t, u_t, dt).squeeze()
            loss = nn.MSELoss()(r_hat, cost)

            self.reward_optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_nets[i].parameters(), 1.0)
            self.reward_optimizers[i].step()

        return loss.item()

    def policy_training(self, batch, ent_coef=1e-3):
        """
        Minimize epigraph Hamiltonian w.r.t. actor via reparameterization.

        H_epi(x,z,px,pz,u) = px^T f(x,u) - pz * ( l_c(x,u) + ln(gamma)*z )

        Actor objective (discrete approximation of integral):
            loss_actor = E[ H_epi * dt ] - ent_coef * Entropy

        Notes:
          - This implementation currently uses V_tilde = V_ret (no explicit z / constraint branch).
          - pz is fixed to -1 (value branch) as in the original code.
        """
        Float = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        states, actions, next_states, rewards, dts, returns = batch

        # Pack states
        x_t = torch.stack(states).type(Float)   # [B, N, D]
        B, N, D = x_t.shape
        x_flat = x_t.view(B, -1)                # [B, N*D]
        device = x_flat.device

        # Generate actions via current policy (reparameterization)
        actions_list, ent_list = [], []
        for i in range(self.n_agents):
            # Step 1: compute px and pz from V_tilde (treated as constants for actor update)
            x_req = x_flat.clone().detach().requires_grad_(True)
            dt = torch.stack(dts).type(Float).view(B, 1)
            log_gamma = torch.log(torch.as_tensor(self.args.gamma, device=device, dtype=dt.dtype))

            # Start from logged actions; will be overwritten for agent i
            whole_actions = torch.stack(actions).type(Float).view(B, -1)  # [B, N*A]

            with torch.no_grad():
                Vl = self.value_nets[i](x_flat)  # [B,1]

            # Epigraph surrogate (kept as provided): V_tilde = V_ret
            V_tilde = Vl

            # Compute px = ∇_x V (detach before using in actor loss)
            x_req = x_flat.clone().detach().requires_grad_(True)
            Vl_for_grad = self.value_nets[i](x_req)
            px = torch.autograd.grad(
                Vl_for_grad.sum(), x_req, create_graph=False, retain_graph=False
            )[0].detach()

            # pz selection (fixed to -1 for value branch)
            pz = -1

            # Step 2: sample action for agent i using reparameterization
            obs_i = x_t[:, i, :]                      # [B, D]
            pi_in = torch.cat([obs_i, dt], dim=-1)     # policy input: [obs, dt]
            mean = self.policy_nets[i](pi_in)          # [B, A]
            dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=self.cov_matrix)
            a_i = dist.rsample()

            # Inject sampled action into the joint action vector
            whole_actions[:, i * self.n_actions:(i + 1) * self.n_actions] = a_i

            actions_list.append(a_i)
            ent_list.append(dist.entropy().view(B, 1))

            # Step 3: build f(x,u) and l_c(x,u)
            # dynamics_nets predicts x_{t+1}, so f_hat ≈ (x_next_pred - x)/dt
            x_next_pred = self.dynamics_nets[i](x_flat, whole_actions, dt).view(B, -1)
            f_hat = (x_next_pred - x_flat) / dt

            # reward_nets trained to predict l_c (here: (-reward) / cost rate)
            l_c_hat = self.reward_nets[i](x_flat, whole_actions, dt).view(B, 1)

            # Step 4: Hamiltonian and actor loss
            H = (px * f_hat).sum(dim=1, keepdim=True) - pz * l_c_hat + log_gamma * V_tilde
            actor_loss = (H * dt).mean()

            self.policy_optimizers[i].zero_grad()
            self.policy_nets[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_nets[i].parameters(), 1.0)
            self.policy_optimizers[i].step()

        return float(actor_loss.detach().item())
