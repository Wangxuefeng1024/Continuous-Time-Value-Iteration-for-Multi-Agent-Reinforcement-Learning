import numpy as np
from continuous_env.multiagent.core import World, Agent, Landmark
from continuous_env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, seed):
        world = World(seed)
        world.dim_c = 2
        world.collaborative = True

        # ===== 1 个 agent =====
        num_agents = 1
        self.num_agents = num_agents
        world.agents = [Agent() for _ in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = False      # 没有其它实体要碰撞了，可以关掉
            agent.silent = True
            agent.size = 0.02
            agent.accel = 3.0         # 看你 core 里默认多少，不想改就注释掉
            agent.max_speed = 1.0

        # ===== 1 个 landmark 作为 goal =====
        world.landmarks = [Landmark()]
        goal = world.landmarks[0]
        goal.name = 'goal'
        goal.collide = False
        goal.movable = False
        goal.size = 0.04
        goal.color = np.array([0.25, 0.95, 0.25])  # 绿色目标

        # 初始化状态
        self.reset_world(world)
        return world

    def reset_world(self, world):
        """
        把 agent 放在下面，把 goal 放在上面，中间一条直线。
        """
        # 起点 / 终点位置（你可以按自己习惯改）
        start_pos = np.array([0.0, -0.5], dtype=np.float32)
        goal_pos = np.array([0.0,  0.5], dtype=np.float32)

        # agent 初始状态
        agent = world.agents[0]
        agent.state.p_pos = start_pos.copy()
        agent.state.p_vel = np.zeros(2, dtype=np.float32)
        agent.state.c = np.zeros(world.dim_c, dtype=np.float32)
        agent.color = np.array([0.35, 0.35, 0.85])

        # goal 位置
        goal = world.landmarks[0]
        goal.state.p_pos = goal_pos.copy()
        goal.state.p_vel = np.zeros(2, dtype=np.float32)
        goal.color = np.array([0.25, 0.95, 0.25])

        # 方便 observation 用
        self.goal_pos = goal_pos

    def reward(self, agent, world):
        """
        simple：负的到目标的欧氏距离。
        后面你想加速度惩罚、时间惩罚都可以往里加。
        """
        goal_pos = world.landmarks[0].state.p_pos
        dist = np.linalg.norm(agent.state.p_pos - goal_pos)
        rew = -dist
        return rew, 0, 0

    def observation(self, agent, world):
        """
        obs = [v_x, v_y, p_x, p_y, goal_x - p_x, goal_y - p_y]
        => obs_dim = 6
        """
        goal_pos = world.landmarks[0].state.p_pos
        rel_goal = goal_pos - agent.state.p_pos

        return np.concatenate(
            [
                agent.state.p_vel,       # 2
                agent.state.p_pos,       # 2
                rel_goal,                # 2
            ],
            axis=0,
        )