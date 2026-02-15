import numpy as np
# from ..core import World, Agent, Landmark
# from ..scenario import BaseScenario
from itertools import permutations

from continuous_env.multiagent.core import World, Agent, Landmark
from continuous_env.multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
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
            agent.size = 0.15
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
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
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
        for l in world.landmarks:
            dists = [np.sum(np.square(a.state.p_pos - l.state.p_pos)) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
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

    def generate_all_goal_states(self, world):
        """
        Generate all possible goal states where each agent ends up on a unique landmark with zero velocity.
        Output: List of numpy arrays, each of shape [n_agents, obs_dim]
        """
        n_agents = len(world.agents)
        n_landmarks = len(world.landmarks)
        assert n_agents == n_landmarks, "Number of agents must equal number of landmarks!"

        goal_states = []

        landmark_positions = [lm.state.p_pos.copy() for lm in world.landmarks]
        all_perms = list(permutations(landmark_positions, n_agents))

        for perm in all_perms:
            obs_goal = []
            for i, agent in enumerate(world.agents):
                v = np.zeros(2)                       # zero velocity
                pos = perm[i]                         # position on a landmark
                entity_pos = [lm - pos for lm in landmark_positions]  # landmark rel pos
                other_pos = [
                    perm[j] - pos for j in range(n_agents) if j != i
                ]                                     # relative other agent pos
                comm = [np.zeros(2) for _ in range(n_agents - 1)]
                obs = np.concatenate([v, pos] + entity_pos + other_pos + comm)
                obs_goal.append(obs)
            goal_states.append(np.stack(obs_goal))     # shape: [n_agents, obs_dim]

        return goal_states  # List of shape [6 x (n_agents, obs_dim)]