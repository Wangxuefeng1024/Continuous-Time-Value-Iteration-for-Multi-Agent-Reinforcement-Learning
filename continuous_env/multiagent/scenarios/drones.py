import numpy as np
from continuous_env.multiagent.core import World, Agent, Landmark
from continuous_env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, seed):
        world = World(seed)
        world.dim_c = 2
        world.collaborative = True

        # ===== 3D =====
        world.dim_p = 3

        # ===== 多 agent =====
        num_agents = 3          # <<< 你想几个 agent 就改这里
        self.num_agents = num_agents
        world.agents = [Agent() for _ in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = f"agent {i}"
            agent.collide = False         # 只做导航先关掉碰撞，训练更稳
            agent.silent = True
            agent.size = 0.02
            agent.accel = 3.0
            agent.max_speed = 1.0

            # 3D 动作/通讯占位，避免 shape 问题
            agent.action.u = np.zeros(world.dim_p, dtype=np.float32)
            agent.action.c = np.zeros(world.dim_c, dtype=np.float32)

        # ===== 共享 goal =====
        world.landmarks = [Landmark()]
        goal = world.landmarks[0]
        goal.name = "goal"
        goal.collide = False
        goal.movable = False
        goal.size = 0.04
        goal.color = np.array([0.25, 0.95, 0.25], dtype=np.float32)

        self.reset_world(world)
        return world

    def reset_world(self, world):
        """
        多 agent 起点不同（原点附近散开），共享一个 3D 目标点
        """
        # 目标点候选（你可以加更多做 transfer）
        candidate_goals = [
            np.array([-0.8,  0.8,  0.4], dtype=np.float32),
            np.array([-0.8, -0.8,  0.4], dtype=np.float32),
            np.array([ 0.8,  0.8,  0.4], dtype=np.float32),
        ]
        goal_pos = candidate_goals[np.random.choice(len(candidate_goals))]

        # goal state
        goal = world.landmarks[0]
        goal.state.p_pos = goal_pos.copy()
        goal.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)

        # agent states：在原点附近随机散开，避免重叠
        # （范围可以调大/调小）
        spawn_low, spawn_high = -0.2, 0.2
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.random.uniform(
                low=spawn_low, high=spawn_high, size=world.dim_p
            ).astype(np.float32)
            agent.state.p_vel = np.zeros(world.dim_p, dtype=np.float32)
            agent.state.c = np.zeros(world.dim_c, dtype=np.float32)
            agent.color = np.array([0.35, 0.35, 0.85], dtype=np.float32)

        self.goal_pos = goal_pos

    # ======== 协作 reward：所有 agent 共享同一个 reward ========
    def reward(self, agent, world):
        """
        全局 reward = -mean_i ||p_i - goal||
        （每个 agent 都返回同一个值，协作最稳）
        """
        goal_pos = world.landmarks[0].state.p_pos
        dists = [np.linalg.norm(a.state.p_pos - goal_pos) for a in world.agents]
        mean_dist = float(np.mean(dists))

        # 越界惩罚：有一个 agent 越界就罚（你也可以改成按个体罚）
        wall_penalty = 0.0
        for a in world.agents:
            if np.any(np.abs(a.state.p_pos) > 1.0):
                wall_penalty = -10.0
                break

        rew = -mean_dist + wall_penalty
        return rew, 0, 0

    def observation(self, agent, world):
        """
        obs = [self_v(3), self_p(3), rel_goal(3), other_rel_p, other_rel_v]
        - other_rel_p: (N-1)*3
        - other_rel_v: (N-1)*3
        总维度 = 9 + 6*(N-1)
        """
        goal_pos = world.landmarks[0].state.p_pos
        rel_goal = goal_pos - agent.state.p_pos

        other_rel_pos = []
        other_rel_vel = []
        for other in world.agents:
            if other is agent:
                continue
            other_rel_pos.append(other.state.p_pos - agent.state.p_pos)
            other_rel_vel.append(other.state.p_vel - agent.state.p_vel)

        if len(other_rel_pos) > 0:
            other_rel_pos = np.concatenate(other_rel_pos, axis=0)
            other_rel_vel = np.concatenate(other_rel_vel, axis=0)
        else:
            other_rel_pos = np.zeros(0, dtype=np.float32)
            other_rel_vel = np.zeros(0, dtype=np.float32)

        obs = np.concatenate(
            [
                agent.state.p_vel,    # 3
                agent.state.p_pos,    # 3
                rel_goal,             # 3
                other_rel_pos,        # 3*(N-1)
            ],
            axis=0,
        ).astype(np.float32)

        return obs