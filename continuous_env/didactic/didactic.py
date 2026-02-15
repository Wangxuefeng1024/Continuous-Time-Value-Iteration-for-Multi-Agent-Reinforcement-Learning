import numpy as np
from qpsolvers import solve_qp   # pip install qpsolvers
from scipy.optimize import minimize      # SciPy ≥1.9 推荐

def sqrtm_psd(M):
    """
    Matrix square root for a real, symmetric positive-(semi)definite matrix.
    Uses eigen-decomposition:  M = Q Λ Qᵀ  →  M¹ᐟ² = Q √Λ Qᵀ.
    """
    eigval, eigvec = np.linalg.eigh(M)
    # numerical safety: clamp tiny negatives caused by round-off
    eigval[eigval < 0.0] = 0.0
    return eigvec @ np.diag(np.sqrt(eigval)) @ eigvec.T


class TwoAgentLQRConstrained:
    """
    Minimal 2-agent linear-quadratic environment with a hard
    separation constraint  c(x) = ||x1 − x2||² − d_safe² ≤ 0.

    State  x ∈ ℝ^{2d}  (concatenate agent-1 and agent-2 positions)
    Action u ∈ ℝ^{2d}  (concatenate their control inputs)
    Dynamics     :  ẋ = u              (continuous)  ->  x_{k+1} = x_k + u Δt
    Running cost :  xᵀ Q x + uᵀ R u
    """

    def __init__(
        self,
        d: int = 1,            # position dimension per agent (1-D is easiest to plot)
        dt: float = 0.05,      # integration / control step
        horizon: int = 30,    # max steps per episode
        q_p: float = 1.0,      # pull to origin
        q_c: float = 1.0,      # coupling strength (||x1−x2||²)
        r: float = 0.1,        # control effort weight
        d_safe: float = 1.0,   # minimal allowed distance
        umax: float = 1.0,  # clip actions element-wise if not None
        seed: int = 111,
    ):
        self.d, self.horizon = d, horizon
        self.rng = np.random.default_rng(seed)

        # ----- quadratic cost matrices -----
        I_d = np.eye(d)
        # coupling block matrix C =
        # [ I  -I ]
        # [ -I  I ]
        C = np.block([[I_d, -I_d],
                      [-I_d, I_d]])

        self.Q = q_p * np.eye(2 * d) + q_c * C
        self.R = r * np.eye(2 * d)

        # analytic Riccati solution for A = 0, B = I
        self.P = np.sqrt(r) * sqrtm_psd(self.Q)           # value matrix
        self.K = (1.0 / np.sqrt(r)) * sqrtm_psd(self.Q)   # optimal feedback

        self.d_safe2 = d_safe ** 2
        self.umax = umax
        self.reset()

    # ========= ground-truth helpers =========
    def V_true(self, x: np.ndarray) -> float:
        """Quadratic value function  V(x) = xᵀ P x ."""
        return float(x.T @ self.P @ x)

    def gradV_true(self, x: np.ndarray) -> np.ndarray:
        """∇V(x) = 2 P x ."""
        return 2.0 * (self.P @ x)

    # def u_star(self, x: np.ndarray) -> np.ndarray:
    #     """Optimal control  u* = −K x ."""
    #     return -self.K @ x

    def u_star(self, x, dt, umax=1.0) -> np.ndarray:
        """
        Solve   min || x + dt*u ||^2
            s.t. u1 - u2 <= -(x1-x2)/dt
                    |u_i| <= umax
        """
        # --- QP matrices -------------------------------------------------------
        H = 2 * (dt ** 2) * np.eye(2)   # (2×2)  positive-definite
        q = 2 * dt * x.astype(float)    # (2,)

        gap = x[0] - x[1]
        G = np.array([[1.0, -1.0]])     # (1×2)
        h = np.array([-gap / dt])       # (1,)

        lb = -umax * np.ones(2)
        ub =  umax * np.ones(2)
        gap    = x[0] - x[1]
        # --- solve ------------------------------------------------------------
        u = solve_qp(H, q, G, h, lb=lb, ub=ub, solver="osqp")
        gap_n  = gap + dt*(u[0]-u[1])
        # print(f"gap={gap:+.4e}, dt={dt:.3e}, "
        #     f"u=({u[0]:+.3f},{u[1]:+.3f}), gap_next={gap_n:+.4e}")
        if u is None:
            raise RuntimeError("QP infeasible: dt too small or umax too tight")
        return u.astype(float)
        
    def c_val(self, x, tol = 1e-6):
        """
        Ordering-constraint value  c(x) = x1 − x2
        ------------------------------------------------
        ≤ 0  → safe   (agent-1 is at or left of agent-2)
        > 0  → violated (agent-1 has overtaken agent-2)
        Works for any `d`:
        • we test only the first coordinate of each agent;
            additional coordinates are ignored for ordering.
        """
        x1_first = x[0]                     # first coord of agent-1
        x2_first = x[1]   # agent-2
        gap = float(x1_first - x2_first)

        # ---- 数值容差: |gap| < tol 视为 0 -------------------------------------
        if abs(gap) < tol:
            gap = 0.0
        return gap

    def c_val_smooth(self, x, beta=5.0):
        gap = x[0] - x[1]
        s   = 1.0 / (1.0 + np.exp(-beta * gap))   # standard sigmoid ∈ (0,1)
        return 2.0 * s - 1.0                      # 线性映射到 (-1,1)

    def V_epi(self, x: np.ndarray, z: float = 0.0) -> float:
        """Epigraph value  ˜V = max{ c(x),  V(x) − z } (single-shot version)."""
        return max(self.c_val(x), self.V_true(x) - z)

    # ========= environment core =========
    # def reset(self):
    #     """
    #     Agent 1 starts uniformly in (-1, 0);  Agent 2 starts uniformly in (0, 1).
    #     Works for d = 1.  (If d > 1, extra coordinates are set to 0.)
    #     """
    #     x1 = -self.rng.uniform(0.0, 1.0)          #  in (-1, 0)
    #     x2 =  self.rng.uniform(0.0, 1.0)          #  in ( 0, 1)
    #     # 2-by-2 matrix form ─ each inner list is a row
    #     self.x = np.array([[x1, x2],      # first row
    #                     [x2, x1]],     # second row
    #                     dtype=float)

    #     self.global_state = np.array([x1, x2],     # second row
    #                     dtype=float)
    #     self.t = 0
    #     return self.x.copy()

    def reset(self):
        """
        Agent 1 starts uniformly in (-1, 0);  Agent 2 starts uniformly in (0, 1).
        Works for d = 1.  (If d > 1, extra coordinates are set to 0.)
        """
        x1 = -0.5          #  in (-1, 0)
        x2 =  0.5        #  in ( 0, 1)
        # 2-by-2 matrix form ─ each inner list is a row
        self.x = np.array([[x1, x2],      # first row
                        [x2, x1]],     # second row
                        dtype=float)

        self.global_state = np.array([x1, x2],     # second row
                        dtype=float)
        self.t = 0
        return self.x.copy()

    def step(self, u, dt):
        """
        One Euler step with action clipping.
        Returns: next_state, reward (negative cost), done, info
        """
        u = u.astype(float)
        if self.umax is not None:
            u = np.clip(u, -self.umax, self.umax).squeeze()

        # integrate the 1D global state
        self.global_state = self.global_state + u * dt

        # update the 2x2 observation matrix (for agent-local view, optional)
        x1, x2 = self.global_state
        self.x = np.array([[x1, x2],
                        [x2, x1]], dtype=float)

        # ----- constraint check & penalty -----
        violated = self.c_val(self.global_state) > 0.0  # constraint: x1 - x2 > 0
        penalty = 10.0 if violated else 0.0
        smooth_penalty = self.c_val_smooth(self.global_state)  # smooth version
        # compute cost using global_state
        dist_cost  = (x1**2 + x2**2)                    # 距离原点平方
        effort_cost = u[0]**2 + u[1]**2                  # 控制能量
        cost = dist_cost + effort_cost    
        reward = [-float(cost)*dt, -penalty]

        # episode bookkeeping
        self.t += 1
        done = self.t >= self.horizon

        # # diagnostics
        # info = {
        #     "constraint": self.c_val(self.global_state),
        #     "V_true": self.V_true(self.global_state),
        # }
        return self.x.copy(), reward, done, smooth_penalty


# ---------------- demo ----------------
if __name__ == "__main__":
    env = TwoAgentLQRConstrained(d=1, seed=42)
    obs = env.reset()
    total_reward = 0.0

    for _ in range(10):
        # greedy optimal action for demo
        action = env.u_star(obs)
        obs, rew, done, info = env.step(action)
        total_reward += rew
        print(f"x={obs},  reward={rew:+.4f},  c(x)={info['constraint']:+.3f}")
        if done:
            break
    print("episode return:", total_reward)


class CoupledOscillatorEnv:
    def __init__(self):
        # System parameters
        self.k = 1.0         # spring constant
        self.b = 0.5         # damping coefficient
        self.lambda_c = 2.0  # coupling strength
        self.beta = 0.01     # control penalty

        # Time step
        self.max_steps = 30
        self.step_count = 0

        # Action space
        self.u_max = 10.0
        self.violation_low = 1.1
        self.violation_high = 1.3
        self.s = 20.0  # 平滑度参数
        # Internal state: [x1, v1, x2, v2]
        self._true_state = None

    def set_reset_points(self, init_states: np.ndarray):
        """
        设置一组初始状态，在 reset 时依次使用。
        :param init_states: shape (N, 4)，对应 [x1, v1, x2, v2]
        """
        self._init_states = init_states
        self._reset_index = 0

    def reset(self):
        # x1, x2 从 [0.9, 1.1] 区间均匀采样；v1, v2 设为 0
        x1 = 1.0
        x2 = 1.0
        v1 = 0.0
        v2 = 0.0
        self._true_state = np.array([x1, v1, x2, v2])
        self.step_count = 0
        return self._get_stacked_state()

    def smooth_interval_sigmoid(self, x, low, high, s=20):
        """
        平滑检测 x 是否在 [low, high] 区间内
        输出范围大约在 (-1, 1)，区间内接近 +1，区间外接近 -1
        """
        # 两个sigmoid相减，得到区间窗口
        in_low  = 1 / (1 + np.exp(-s * (x - low)))
        in_high = 1 / (1 + np.exp(-s * (high - x)))
        inside = in_low * in_high  # x在区间内时接近1，否则接近0

        # 映射到 (-1,1)，区间内≈+1，外部≈-1
        return 2 * inside - 1

    # def smooth_violation(self, x1, x2):
    #     v1 = self.smooth_interval_sigmoid(x1, self.violation_low, self.violation_high, self.s)
    #     v2 = self.smooth_interval_sigmoid(x2, self.violation_low, self.violation_high, self.s)
    #     return v1 + v2
    
    def smooth_violation(self, x1, x2, s=20):
        raw = x1 - x2 +0.02
        sigmoid_val = 1 / (1 + np.exp(-s * raw))
        # Scale and shift: from (0, 1) → (-1, 10)
        smooth_penalty = sigmoid_val * 1 - (1 - sigmoid_val)
        return smooth_penalty
    
    
    def step(self, action, dt=None):
        if dt is None:
            dt = self.dt

        x1, v1, x2, v2 = self._true_state

        # Expect normalized action in [-1, 1]
        u1, u2 = np.clip(action, -1.0, 1.0) * self.u_max

        # Dynamics
        a1 = -self.k * x1 - self.b * v1 + u1
        a2 = -self.k * x2 - self.b * v2 + u2

        # Euler integration
        v1 += a1 * dt
        x1 += v1 * dt
        v2 += a2 * dt
        x2 += v2 * dt

        self._true_state = np.array([x1, v1, x2, v2])
        self.step_count += 1

        # Smoothed reward
        cost = (
            x1**2 + x2**2 +
            self.lambda_c * (x1 - x2)**2 +
            self.beta * (u1**2 + u2**2)
        )

        penalty = 0.0
        # if (x1 > self.violation_low and x1 < self.violation_high) or (x2 > self.violation_low and x2 < self.violation_high):
        #     penalty = -10.0  # 惩罚分值，可以调节
        if x1 > x2 + 0.02:
            penalty = -10.0  # 惩罚分值，可以调节
        reward = -cost / 30.0  # scale down to smooth range
        s = 20.0  # 平滑度调节参数
        raw = x1 - x2
        sigmoid_val = 1 / (1 + np.exp(-s * raw))
        smooth_violation_val = self.smooth_violation(x1, x2)

        done = self.step_count >= self.max_steps
        return self._get_stacked_state(), [reward, penalty], done, smooth_violation_val

    def _get_agent_observation(self, agent_id):
        x1, v1, x2, v2 = self._true_state
        if agent_id == 0:
            return np.array([x1, v1, x2, v2]).flatten()
        elif agent_id == 1:
            return np.array([x2, v2, x1, v1]).flatten()
        else:
            raise ValueError("Invalid agent_id. Use 0 or 1.")

    def _get_stacked_state(self):
        obs1 = self._get_agent_observation(0)
        obs2 = self._get_agent_observation(1)
        return np.stack([obs1, obs2])  # shape: [2, 4]

    def get_agent_observations(self):
        return [self._get_agent_observation(0), self._get_agent_observation(1)]

    def seed(self, seed=None):
        np.random.seed(seed)


    # ===== Ground-truth via Augmented Lagrangian LQR =====
    # ===== Build system matrices and LQR controller =====
    def _sys_mats(self):
        """Continuous-time linearized dynamics xdot = A x + B u; quadratic cost x'Qx + u'Ru."""
        k, b = self.k, self.b
        lam_c, beta = self.lambda_c, self.beta

        # x=[x1,v1,x2,v2], u=[u1,u2]
        A = np.array([[0,   1,   0,   0],
                    [-k, -b,   0,   0],
                    [0,   0,   0,   1],
                    [0,   0,  -k,  -b]], dtype=float)
        B = np.array([[0,0],
                    [1,0],
                    [0,0],
                    [0,1]], dtype=float)

        # stage cost: x1^2 + x2^2 + lam_c (x1-x2)^2 + beta ||u||^2
        Q = np.array([[1+lam_c, 0,        -lam_c, 0],
                    [0,       0,         0,     0],
                    [-lam_c,  0,         1+lam_c, 0],
                    [0,       0,         0,     0]], dtype=float)
        R = beta * np.eye(2)
        return A, B, Q, R

    def _make_lqr(self):
        A, B, Q, R = self._sys_mats()
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)  # R^{-1} B^T P
        self._A, self._B, self._Q, self._R = A, B, Q, R
        self._P, self._K = P, K

    def set_gt_controller(self, alpha_barrier: float = 5.0, use_R_metric: bool = True, clip_to_umax: bool = True):
        """
        Prepare the 'ground-truth' controller:
        - LQR for performance
        - CBF (barrier) projection for hard constraint x1 - x2 - 0.02 <= 0
        alpha_barrier: class-K parameter in ∂h/∂x (Ax+Bu) + α h ≥ 0
        use_R_metric:  project using (u - u_LQR)^T R (u - u_LQR); else Euclidean.
        """
        self._make_lqr()
        self._gt_alpha = float(alpha_barrier)
        self._gt_clip = bool(clip_to_umax)
        self._proj_use_R = bool(use_R_metric)

    def _h_and_jac(self, x):
        """
        Barrier h(x) = 0.02 - (x1 - x2)  (>= 0 is safe).
        ∇h = [-1, 0, +1, 0]^T
        """
        x1, v1, x2, v2 = x
        h = 0.02 - (x1 - x2)
        dhdx = np.array([-1.0, 0.0, 1.0, 0.0])
        return h, dhdx

    def _cbf_project(self, x, u_lqr):
        """
        Solve:  minimize 1/2 (u - u_lqr)^T W (u - u_lqr)
                subject to  dhdx^T (A x + B u) + α h >= 0
        Closed-form projection onto a single half-space:
        If a^T u_lqr >= b  ⇒  return u_lqr
        else u* = u_lqr + τ W^{-1} a, with τ = (b - a^T u_lqr) / (a^T W^{-1} a)
        """
        A, B, R = self._A, self._B, self._R
        W = R if self._proj_use_R else np.eye(2)

        h, dhdx = self._h_and_jac(x)
        a = (dhdx @ B).reshape(-1)          # shape (2,)
        b = - (dhdx @ (A @ x)) - self._gt_alpha * h

        # Check constraint at u_lqr
        aTu = float(a @ u_lqr)
        if aTu >= b:
            u = u_lqr.copy()
        else:
            # Project onto the active plane a^T u = b in the W-metric
            Winv = np.linalg.inv(W)
            denom = float(a @ (Winv @ a))
            if denom <= 1e-12:
                # Degenerate (shouldn't happen for full-rank W, B)
                u = u_lqr.copy()
            else:
                tau = (b - aTu) / denom
                u = u_lqr + (Winv @ a) * tau

        if self._gt_clip:
            u = np.clip(u, -self.u_max, self.u_max)
        return u

    def optimal_action_gt(self, x=None):
        """
        Ground-truth action = LQR action filtered by a CBF projection.
        """
        if not hasattr(self, "_K"):
            # Backward compatibility: names if set_gt_controller hasn't been called
            self.set_gt_controller()

        if x is None:
            if self._true_state is None:
                raise RuntimeError("Call reset() first or pass x explicitly.")
            x = self._true_state
        x = np.asarray(x).reshape(-1)

        # Unconstrained optimal (performance)
        u_lqr = - self._K @ x

        # Safety filter (hard state constraint)
        u = self._cbf_project(x, u_lqr)
        return u

    # (Optional) keep a reference value/grad from LQR for diagnostics in safe region
    def value_gt(self, x=None):
        """
        Quadratic LQR value V=x^T P x (good diagnostic when barrier inactive).
        """
        if not hasattr(self, "_P"):
            self.set_gt_controller()
        if x is None:
            if self._true_state is None:
                raise RuntimeError("Call reset() first or pass x explicitly.")
            x = self._true_state
        x = np.asarray(x).reshape(-1, 1)
        return float(x.T @ self._P @ x)

    def value_grad_gt(self, x=None):
        if not hasattr(self, "_P"):
            self.set_gt_controller()
        if x is None:
            if self._true_state is None:
                raise RuntimeError("Call reset() first or pass x explicitly.")
            x = self._true_state
        x = np.asarray(x).reshape(-1)
        return (2.0 * self._P @ x)
