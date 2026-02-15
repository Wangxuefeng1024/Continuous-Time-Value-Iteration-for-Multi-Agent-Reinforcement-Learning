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
        Agent 始终从 (0,0) 出发。
        Goal 从 [左上, 左下, 右上] 三个点中随机选择。
        """
        # 1. 固定的起始位置 (0, 0)
        start_pos = np.array([0.0, 0.0], dtype=np.float32)

        # 2. 三个候选目标位置 (假设地图范围 -1 到 1，稍微留点边距选 0.8)
        # Task 1: 左上
        # Task 2: 左下
        # Task 3: 右上
        # (Task 4: 右下 - 留作测试，这里不放)
        candidate_goals = [
            np.array([-0.8, 0.8], dtype=np.float32),  # 左上
            np.array([-0.8, -0.8], dtype=np.float32), # 左下
            np.array([0.8, 0.8], dtype=np.float32)    # 右上
        ]

        # 3. 随机选择一个 Goal
        # idx = np.random.randint(0, len(candidate_goals)) # 如果你想完全随机
        # goal_pos = candidate_goals[idx]
        
        # 为了方便你做 Transfer Learning (训练 Task A -> Task B -> Task C)
        # 你可能需要手动指定，或者让环境随机。
        # 这里先写成完全随机，如果你需要手动指定任务，请告诉我，我在 World 里加个 task_id。
        idx = np.random.choice(len(candidate_goals))
        goal_pos = candidate_goals[idx]

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
        """
        goal_pos = world.landmarks[0].state.p_pos
        dist = np.linalg.norm(agent.state.p_pos - goal_pos)
        
        # 加上一点边界惩罚，防止它跑出 (-1, 1) 的地图
        # 虽然这对于 SF 来说可能不是必须的，但对于训练稳定性有好处
        wall_penalty = 0
        if np.any(np.abs(agent.state.p_pos) > 1.0):
            wall_penalty = -10.0

        rew = -dist + wall_penalty
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