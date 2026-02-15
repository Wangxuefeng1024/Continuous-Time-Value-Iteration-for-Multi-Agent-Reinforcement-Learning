import numpy as np
from continuous_env.multiagent.core import World, Agent, Landmark
from continuous_env.multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, seed):
        world = World(seed)
        world.dim_c = 2
        num_agents = 3
        num_goals = 3
        num_walls = 2
        world.collaborative = True
        self.num_agents = num_agents
        self.num_goals = num_goals

        # Create agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'agent {i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.05

        # Create landmarks: 3 goals + 2 corridor walls
        world.landmarks = [Landmark() for _ in range(num_goals + num_walls)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'landmark {i}'
            landmark.size = 0.05 if i < num_goals else 0.30  # wall is larger
            if i < self.num_goals:
                landmark.collide = False
                landmark.movable = False
                landmark.size  = 0.05
                landmark.color = np.array([0.25, 0.95, 0.25])    # green goals
            else:
                landmark.collide = True
                landmark.movable = False
                landmark.size  = 0.30
                landmark.color = np.array([0.8, 0.2, 0.2])       # red obstacles

        self.reset_world(world)
        return world

    def reset_world(self, world):
        area_size = 1.0
        corridor_width = 0.3

        # Agent appearance
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])

        # Landmark appearance
        for i, landmark in enumerate(world.landmarks):
            if i < 3:
                landmark.color = np.array([0.25, 0.95, 0.25])  # green: goals
            else:
                landmark.color = np.array([0.8, 0.2, 0.2])     # red: walls

        # Agents: start on one side of the corridor
        agent_positions = [
            np.array([-0.38, -0.35], dtype=np.float32),
            np.array([0.30, -0.50], dtype=np.float32),
            np.array([0.03, -0.24], dtype=np.float32),
        ]
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = agent_positions[i]
            agent.state.p_vel = np.zeros(2)
            agent.state.c = np.zeros(world.dim_c)

        landmark_positions = [
            np.array([0.30, 0.70], dtype=np.float32),
            np.array([-0.30, 0.65], dtype=np.float32),
            np.array([0.09, 0.82], dtype=np.float32),
        ]
        # Goal landmarks: on the other side of corridor
        for i in range(3):
            world.landmarks[i].state.p_pos = landmark_positions[i]
            world.landmarks[i].state.p_vel = np.zeros(2)

        # Walls: placed at left and right to form corridor
        y_center = 0.15
        world.landmarks[3].state.p_pos = np.array([-0.4, y_center])
        world.landmarks[4].state.p_pos = np.array([ 0.4, y_center])
        for i in [3, 4]:
            world.landmarks[i].state.p_vel = np.zeros(2)
        self.goals = landmark_positions

    def is_collision(self, ent1, ent2):
        delta_pos = ent1.state.p_pos - ent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        min_dist = ent1.size + ent2.size
        return dist < min_dist

    def reward(self, agent, world):
        idx = world.agents.index(agent)
        # goals = self.landmark2goal(world)
        # goal = self.goals[idx % len(self.goals)]
        goal = self.goals

        dist = np.linalg.norm(agent.state.p_pos - goal[idx])
        rew = -dist

        dis_rew = 0
        # for other in world.agents:
        #     if other is agent:
        #         continue
        #     if self.is_collision(agent, other):
        #         dis_rew -= 15
        for obs in world.landmarks[self.num_agents:]:
            if self.is_collision(agent, obs):
                dis_rew = -10
        continuous_rew = self.continuous_penalty(world, agent)

        return rew, dis_rew, continuous_rew

    def continuous_penalty(self, world, agent, collision_scale=20.0, eps=1e-9):
        """
        Continuous obstacle-only penalty.

        For every agent–obstacle pair:
        penetration = (r_agent + r_obs) - distance(agent, obstacle)
        if penetration > 0 (i.e., collision/overlap):
            add collision_scale * penetration
        else:
            add 0

        No negative contributions when no collision; no agent–agent terms.
        """
        idx = world.agents.index(agent)

        total = 0.0
        obstacles = world.landmarks[self.num_agents:]  # obstacles only
        # for agent in world.agents:
        pa = world.agents[idx].state.p_pos
        ra = world.agents[idx].size
        for obs in obstacles:
            po = obs.state.p_pos
            ro = obs.size
            d = float(np.linalg.norm(pa - po) + eps)
            penetration = (ra + ro) - d
            if penetration > 0.0:
                penetration = collision_scale * penetration
            elif penetration <= 0.0:
                penetration = 0.5*penetration
            total += penetration
        return total/ 2.0  # Avoid double counting (i,j) and (j,i)

    def observation(self, agent, world):
        # Goal + wall positions relative to agent
        entity_pos = [ent.state.p_pos - agent.state.p_pos for ent in world.landmarks]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in world.agents if other is not agent]
        return np.concatenate([agent.state.p_vel, agent.state.p_pos] + entity_pos + other_pos )

