import numpy as np
from continuous_env.multiagent.core import World, Agent, Landmark
from continuous_env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, seed):
        world = World(seed)
        world.dim_c = 2
        # collaborative = True 表示 reward 是共享的或者是一起算的，
        # 在 predator-prey 里，如果是单智能体追捕，通常不需要 collaborative，但在 MPE 框架下这通常不影响单智能体逻辑
        world.collaborative = True 

        # ===== 1 个 Agent (Predator / 捕食者) =====
        num_agents = 1
        self.num_agents = num_agents
        world.agents = [Agent() for _ in range(num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = True       # 开启碰撞，这样追上了才有物理接触的感觉
            agent.silent = True
            agent.size = 0.05          # 稍微大一点，方便抓
            agent.accel = 3.0
            agent.max_speed = 1.0      # 捕食者速度

        # ===== 1 个 Landmark (Prey / 猎物) =====
        world.landmarks = [Landmark()]
        goal = world.landmarks[0]
        goal.name = 'prey'
        goal.collide = False       # Landmark 通常设为 False，除非你想模拟物理撞击
        
        # [关键修改 1] 让它可移动
        goal.movable = True        
        
        goal.size = 0.03           # 猎物稍微小一点
        goal.color = np.array([0.85, 0.35, 0.35])  # 红色目标 (Prey)

        self.reset_world(world)
        return world

    def reset_world(self, world):
        """
        每次 reset，Prey 出现在固定的位置，并以固定的速度移动。
        这样环境就是确定性的 (Deterministic)。
        """
        # 1. 设置 Predator (Agent) - 固定在原点
        world.agents[0].state.p_pos = np.array([0.0, 0.0], dtype=np.float32)
        world.agents[0].state.p_vel = np.zeros(2, dtype=np.float32)
        world.agents[0].state.c = np.zeros(world.dim_c, dtype=np.float32)
        world.agents[0].color = np.array([0.35, 0.35, 0.85])

        # 2. 设置 Prey (Landmark)
        prey = world.landmarks[0]
        
        # [修改] 固定出生位置 (例如：固定在右上方)
        prey.state.p_pos = np.array([0.5, 0.5], dtype=np.float32)
        
        # [修改] 固定初速度 (例如：固定往左下角跑)
        # 这样 Agent 必须学会去拦截它
        prey.state.p_vel = np.array([-0.2, -0.2], dtype=np.float32)
        
        prey.color = np.array([0.25, 0.95, 0.25])

    def reward(self, agent, world):
        """
        Reward: 负的距离。
        如果 Prey 跑出界了，可以给额外的惩罚（可选）。
        """
        prey = world.landmarks[0]
        dist = np.linalg.norm(agent.state.p_pos - prey.state.p_pos)
        
        # 基础距离惩罚
        rew = -dist
            
        return rew, 0, 0

    def observation(self, agent, world):
        """
        obs dimension = 8
        包含：
        1. Agent 自己的速度 (2)
        2. Agent 自己的位置 (2)
        3. Agent 到 Prey 的相对位置 (2)
        4. [新增] Prey 的速度 (2) -> 这样 Agent 才能学会预判轨迹
        """
        prey = world.landmarks[0]
        rel_pos = prey.state.p_pos - agent.state.p_pos
        rel_vel = prey.state.p_vel - agent.state.p_vel # 相对速度通常更有用

        return np.concatenate(
            [
                rel_vel,   # [2] Self Velocity
                agent.state.p_pos,   # [2] Self Position
                rel_pos
            ],
            axis=0,
        )