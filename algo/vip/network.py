import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Adaptive tanh(ax) ----------
class AdaptiveTanh(nn.Module):
    """
    tanh(a ⊙ x) with learnable 'a'.
    - If shared=True: a is a scalar shared by all units.
    - Else: a is a vector of length 'features' (per-neuron).
    """
    def __init__(self, features: int, init_a: float = 1.0, shared: bool = False):
        super().__init__()
        if shared:
            self.a = nn.Parameter(torch.tensor(init_a, dtype=torch.float32))
            self.register_buffer("_shape", torch.tensor([-1]))  # sentinel
        else:
            self.a = nn.Parameter(torch.full((features,), init_a, dtype=torch.float32))
            self.register_buffer("_shape", torch.tensor([features]))

        self.shared = shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared:
            return torch.tanh(self.a * x)
        # per-neuron: broadcast across batch/time dims, last dim must match
        return torch.tanh(x * self.a)

 
class DynamicsNet(nn.Module):
    """
    Your original used ReLU everywhere. If you want to embed adaptive tanh,
    here's a version that switches to AdaptiveTanh for hidden layers.
    If you'd rather keep ReLU, just revert to F.relu.
    """
    def __init__(self, obs_dim, act_dim, n_agents, use_adaptive=True, adaptive_shared=False):
        super(DynamicsNet, self).__init__()
        in_dim = ((obs_dim + act_dim) * n_agents) + 1
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, obs_dim * n_agents)

        self.use_adaptive = use_adaptive
        if use_adaptive:
            self.act1 = AdaptiveTanh(128, shared=adaptive_shared)
            self.act2 = AdaptiveTanh(128, shared=adaptive_shared)

    def forward(self, x, u, dt):
        xu = torch.cat([x, u, dt], dim=-1)
        if self.use_adaptive:
            h = self.act1(self.fc1(xu))
            h = self.act2(self.fc2(h))
        else:
            h = F.relu(self.fc1(xu))
            h = F.relu(self.fc2(h))
        return self.fc3(h)

# ========== RewardNet3 ==========
class RewardNet(nn.Module):
    """
    Same idea as DynamicsNet3: embed adaptive tanh in hidden layers.
    """
    def __init__(self, obs_dim, act_dim, n_agents, use_adaptive=True, adaptive_shared=False):
        super(RewardNet, self).__init__()
        in_dim = ((obs_dim + act_dim) * n_agents) + 1
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

        self.use_adaptive = use_adaptive
        if use_adaptive:
            self.act1 = AdaptiveTanh(128, shared=adaptive_shared)
            self.act2 = AdaptiveTanh(128, shared=adaptive_shared)

    def forward(self, x, u, dt):
        xu = torch.cat([x, u, dt], dim=-1)
        if self.use_adaptive:
            h = self.act1(self.fc1(xu))
            h = self.act2(self.fc2(h))
        else:
            h = F.relu(self.fc1(xu))
            h = F.relu(self.fc2(h))
        return self.out(h)
    

class ValueNet(nn.Module):
    def __init__(self, obs_dim, n_agents, relu=False):
        super(ValueNet, self).__init__()
        input_dim = obs_dim * n_agents 
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)
        self.relu = relu

    def forward(self, x_all):
        if self.relu:
            x = F.relu(self.fc1(x_all))
            x = F.relu(self.fc2(x))
        else:
            x = torch.tanh(self.fc1(x_all))
            x = torch.tanh(self.fc2(x))
        return self.out(x)

class PolicyNet(nn.Module):
    """
    Keep the first two layers ReLU (as in your original),
    replace the last hidden tanh and output tanh with AdaptiveTanh.
    """
    def __init__(self, obs_dim, act_dim, adaptive_shared=False):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim + 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, act_dim)

        # adaptive tanh for the last hidden + output
        self.act3 = AdaptiveTanh(64, shared=adaptive_shared)
        self.act_out = AdaptiveTanh(act_dim, shared=adaptive_shared)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.act3(self.fc3(x))
        return self.act_out(self.out(x))