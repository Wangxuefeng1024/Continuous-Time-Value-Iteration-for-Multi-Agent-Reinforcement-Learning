import numpy as np
# from ..core import World, Agent, Landmark
# from ..scenario import BaseScenario
from itertools import permutations

from continuous_env.multiagent.core import World, Agent, Landmark
from continuous_env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, seed):
        world = World(seed)
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        np.random.seed(world.seed)
        agent_pos = np.array([
            [ 0.00, 0.60],
            [-0.60, -0.60],   # agent 0  # agent 1
            [ 0.60, -0.60],   # agent 2
        ], dtype=np.float32)

        obs_pos = np.array([
            [ 0.73, 0.35],   # goal for agent 0
            [ 0.60, 0.09],  # goal for agent 1
            [-0.43, -.35],   # goal for agent 2
        ], dtype=np.float32)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = agent_pos[i].copy()
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = obs_pos[i].copy()
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # continuous-friendly version: no sqrt, smoother
        rew = 0
        dis_rew = 0
        for l in world.landmarks:
            dists = [np.sum(np.square(a.state.p_pos - l.state.p_pos)) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if a is not agent:
                    if self.is_collision(a, agent):
                        dis_rew = -10
        continuous_rew = self.continuous_penalty(world)
        return rew, dis_rew, continuous_rew

    def continuous_penalty(self, world, collision_scale=20.0):
        """
        连续版 penalty：所有 agent 两两距离的和；若发生碰撞（距离 < 半径和），
        则把该对距离乘以一个较大的系数 collision_scale 以夸张惩罚。
        注意：按你的要求用双重循环累加，最后除以 2，避免双计数。
        """
        agents = world.agents
        n = len(agents)
        total = 0.0
        eps = 1e-9
        boundary = agents[0].size * 2.0  # 假设所有 agent 半径相同
        for i in range(n):
            pi = agents[i].state.p_pos
            ri = agents[i].size

            for j in range(n):
                if j == i:
                    continue
                pj = agents[j].state.p_pos
                rj = agents[j].size

                d = float(np.linalg.norm(pi - pj) + eps)  # 距离
                term = boundary - d
                # 若“碰撞”（小于半径和），夸张惩罚
                if d < (ri + rj):
                    term *= collision_scale
                total += term

        return total / 2.0  # 因为(i,j)和(j,i)都算了

    def discontinuous_reward(self, agent, world):
        # continuous-friendly version: no sqrt, smoother
        rew = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
