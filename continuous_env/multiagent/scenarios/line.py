import numpy as np
from continuous_env.multiagent.core import World, Agent, Landmark
from continuous_env.multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, seed):
        world = World(seed)
        world.dim_c = 2
        self.num_agents = 3
        self.num_goals  = 3
        self.num_obs    = 0
        world.collaborative = True

        # === Agents ===
        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name    = f'agent {i}'
            agent.collide = True
            agent.silent  = True
            agent.size    = 0.05
            agent.color   = np.array([0.35, 0.35, 0.85])  # blue

        # === Landmarks: [0..2] goals, [3..4] obstacles ===
        world.landmarks = [Landmark() for _ in range(self.num_goals + self.num_obs)]
        for i, lm in enumerate(world.landmarks):

            if i < self.num_goals:
                lm.collide = False
                lm.movable = False
                lm.size  = 0.03
                lm.color = np.array([0.25, 0.95, 0.25])    # green goals
            else:
                lm.collide = True
                lm.movable = False
                lm.size  = 0.07
                lm.color = np.array([0.8, 0.2, 0.2])       # red obstacles

        self.reset_world(world)
        return world

    def reset_world(self, world):
        # ---------------- 固定位置（无随机） ----------------
        # Agents（起点：三角形）
        agent_pos = np.array([
            [ 0.00, -0.70],   # agent 0：底部中心
            [-0.35, -0.35],   # agent 1：左上
            [ 0.35, -0.35],   # agent 2：右上
        ], dtype=np.float32)

        for i, ag in enumerate(world.agents):
            ag.state.p_pos = agent_pos[i].copy()
            ag.state.p_vel = np.zeros(2, dtype=np.float32)
            ag.state.c     = np.zeros(world.dim_c, dtype=np.float32)

        # Goals：终点一条直线（固定），映射为 agent i -> goal i
        # 这里用一条水平线 y = 0.75，三个点等间距
        goal_pos = np.array([
            [-0.50,  0.75],   # goal 0
            [ 0.00,  0.75],   # goal 1
            [ 0.50,  0.75],   # goal 2
        ], dtype=np.float32)

        for i in range(self.num_goals):
            world.landmarks[i].state.p_pos = goal_pos[i].copy()
            world.landmarks[i].state.p_vel = np.zeros(2, dtype=np.float32)

    # ========================= Reward / Penalty =========================
    def reward(self, agent, world):
        """
        总奖励 = 逼近自己目标的距离奖励 + 离散碰撞惩罚
        其中：dis_rew 只要发生一次碰撞就 -10（每个碰撞对象各 -10）
        不把 continuous penalty 加入总奖励。
        """
        idx = world.agents.index(agent)

        # 距离目标的稀疏型奖励（越近越好）
        goal_pos = world.landmarks[idx].state.p_pos
        dist = float(np.linalg.norm(agent.state.p_pos - goal_pos))
        rew_goal = -dist

        # 离散碰撞惩罚：agent–agent 与 agent–obstacle
        # dis_rew = 0.0
        # # agent–agent
        # for other in world.agents:
        #     if other is agent:
        #         continue
        #     if self.is_collision(agent, other):
        #         dis_rew -= 10.0

        # # agent–obstacle（注意：障碍在 [self.num_goals:]）
        # for lm in world.landmarks[self.num_goals:]:
        #     if self.is_collision(agent, lm):
        #         dis_rew -= 10.0

        # cont_cost = self.continuous_penalty_agent(world, agent, scale_ao=20.0, scale_aa=20.0)

        return rew_goal, 0.0, 0.0

    def continuous_penalty_agent(self, world, agent, scale_ao=20.0, scale_aa=20.0, eps=1e-9):
        """
        对“当前 agent”计算连续碰撞代价（正数）：
        hinge = max(0, (r_i + r_j) - d)
        - agent–obstacle（两处障碍）
        - agent–agent（其它两个智能体）
        """
        pa = agent.state.p_pos
        ra = agent.size
        total = 0.0

        # agent–obstacle
        for lm in world.landmarks[self.num_goals:]:
            po, ro = lm.state.p_pos, lm.size
            d = float(np.linalg.norm(pa - po) + eps)
            hinge = (ra + ro) - d
            if hinge > 0.0:
                total += scale_ao * hinge

        # agent–agent
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
        obs = [ vel(2), pos(2),
                rel_self_goal(2),
                rel_obstacles( num_obs * 2 ),     # 2 * 2
                rel_other_agents( (N-1) * 2 ) ]   # 2 * 2
        """
        idx = world.agents.index(agent)

        # 自己的 goal（固定映射）
        rel_goal = world.landmarks[idx].state.p_pos - agent.state.p_pos

        # 其他 2 个智能体
        rel_others = [
            other.state.p_pos - agent.state.p_pos
            for other in world.agents if other is not agent
        ]

        parts = [agent.state.p_vel, agent.state.p_pos, rel_goal] + rel_others
        return np.concatenate(parts, axis=0).astype(np.float32)

    # ========================= Utils =========================
    def is_collision(self, e1, e2):
        delta = e1.state.p_pos - e2.state.p_pos
        dist  = np.sqrt((delta**2).sum())
        return dist < (e1.size + e2.size)
