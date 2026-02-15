import numpy as np
from continuous_env.multiagent.core import World, Agent, Landmark
from continuous_env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, seed=None):
        world = World(seed)
        world.dim_c = 2
        world.collaborative = False

        # === 只要3个predators，不把prey算进agents ===
        self.num_pred = 3
        self.num_agents = self.num_pred

        # === 创建 predators ===
        world.agents = [Agent() for _ in range(self.num_pred)]
        for i, ag in enumerate(world.agents):
            ag.name = f'predator {i}'
            ag.silent = True
            ag.collide = True               # predator之间有物理碰撞
            ag.size = 0.05
            ag.accel = 3.5                  # 比goal稍快
            ag.max_speed = 1.2
            ag.color = np.array([0.35, 0.85, 0.35], dtype=np.float32)

        # === 创建“可移动目标”goal（使用一个 landmark 表示） ===
        self.goal_idx = 0
        world.landmarks = [Landmark() for _ in range(1)]  # 只有一个移动目标
        goal = world.landmarks[self.goal_idx]
        goal.name = "goal"
        goal.size = 0.05
        goal.collide = False              # 不参与碰撞，不会把predator弹开
        goal.movable = False              # 由我们脚本移动（不走物理引擎力学）
        goal.boundary = False
        goal.color = np.array([0.95, 0.80, 0.25], dtype=np.float32)  # 黄橙色

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # 固定 predator 初始位置（你给的坐标）
        pred_pos = np.array([
            [ 0.00,  0.60],   # predator 0
            [-0.60, -0.60],   # predator 1
            [ 0.60, -0.60],   # predator 2
        ], dtype=np.float32)

        for i, ag in enumerate(world.agents):
            ag.state.p_pos = pred_pos[i].copy()
            ag.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)
            ag.state.c     = np.zeros(world.dim_c, dtype=np.float32)

        # 固定 goal 初始位置（你给的坐标）
        goal = world.landmarks[self.goal_idx]
        goal.state.p_pos = np.array([0.25, 0.40], dtype=np.float32)
        goal.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)

    # ========= 每步先调用：让goal随机移动（但“不要太快”） =========
    def step_goal_random(self, world, noise_std=0.4, max_step=0.03, bounds=1.0):
        """
        在 env 的每个 step、调用 world.step() 之前先调用本函数：
            scenario.step_goal_random(world)
        - noise_std 控制随机方向的抖动
        - max_step 控制单步位移上限（速度上限，越小越慢）
        - bounds   保持在 [-bounds, +bounds] 里
        """
        goal = world.landmarks[self.goal_idx]

        # 采样随机位移方向
        d = np.random.normal(0.0, noise_std, size=2).astype(np.float32)
        norm = np.linalg.norm(d) + 1e-9
        d = d / norm * max_step   # 归一化到单步最大步长

        # 更新位置并裁边界
        new_pos = goal.state.p_pos + d
        new_pos = np.clip(new_pos, -bounds, bounds)
        goal.state.p_pos = new_pos
        goal.state.p_vel = d  # 仅用于观测/可视化（不走力学）

    # ========================= Reward =========================
    def reward(self, agent, world):
        """
        predator 的奖励：
            - 与 goal 的负距离（越近越好）
            - 离散碰撞惩罚：只对 predator–predator，每对碰撞 -10
        不把“连续罚项”加到总奖励里（你之前的要求），
        若需要记录连续罚项，可单独调用 continuous_penalty_agent(...)。
        返回 (total_rew, dis_pen) 方便日志。
        """
        # 这里只会对 predator 调用；若被误调用到其它实体，返回0
        if agent not in world.agents:
            return 0.0, 0.0

        goal = world.landmarks[self.goal_idx]

        # 逼近目标
        dist = float(np.linalg.norm(agent.state.p_pos - goal.state.p_pos))
        rew_goal = -dist

        # predator–predator 离散碰撞惩罚
        dis_pen = 0.0
        for other in world.agents:
            if other is agent:
                continue
            if self.is_collision(agent, other):
                dis_pen -= 10.0

        cont_pen = self.continuous_penalty_agent(world, agent, scale_aa=20.0)
        return rew_goal, dis_pen, cont_pen

    # ========================= Continuous penalty（仅pred-pred） =========================
    def continuous_penalty_agent(self, world, agent, scale_aa=20.0, eps=1e-9):
        """
        仅统计 predator–predator 的连续重叠代价（正数），不包含 goal：
            hinge = max(0, (r_i + r_j) - d)
        你可以在训练里单独调用这个函数并记录日志；不并入 total reward。
        """
        if agent not in world.agents:
            return 0.0

        pa = agent.state.p_pos
        ra = agent.size
        total = 0.0

        for other in world.agents:
            if other is agent:
                continue
            pb, rb = other.state.p_pos, other.size
            d = float(np.linalg.norm(pa - pb) + eps)
            hinge = (ra + rb) - d
            if hinge > 0.0:
                total += scale_aa * hinge

        return total

    # ========================= Observation =========================
    def observation(self, agent, world):
        """
        predator 观测：
          [ self_vel(2), self_pos(2),
            rel_goal_pos(2),
            rel_other_preds( (num_pred-1) * 2 ) ]
        """
        self.step_goal_random(world)  # 每步先让 goal 随机移动
        goal = world.landmarks[self.goal_idx]
        rel_goal = goal.state.p_pos - agent.state.p_pos
        rel_others = [
            other.state.p_pos - agent.state.p_pos
            for other in world.agents if other is not agent
        ]
        parts = [agent.state.p_vel, agent.state.p_pos, rel_goal] + rel_others
        return np.concatenate(parts, axis=0).astype(np.float32)

    # ========================= Utils =========================
    def is_collision(self, a, b):
        delta = a.state.p_pos - b.state.p_pos
        dist = np.sqrt((delta**2).sum())
        return dist < (a.size + b.size)

    # 训练端：哪些是可训练体（predators）
    def trainable_agents(self, world):
        return list(world.agents)
