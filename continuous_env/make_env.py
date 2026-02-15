# ======= make_env.py (support dt/substeps setup) =======
from continuous_env.multiagent.environment import MultiAgentEnv
from continuous_env.multiagent.scenarios import load

def make_env(scenario_name, seed):
    # load scenario from script
    scenario = load(scenario_name).Scenario()
    # create world
    world = scenario.make_world(seed)
    # create multiagent environment
    env = MultiAgentEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=getattr(scenario, "info", None),
        done_callback=getattr(scenario, "done", None),
        shared_viewer=True
    )
    # override continuous-time config
    return env


# ======= example usage =======
if __name__ == "__main__":
    env = make_env("simple_spread", dt=0.02, substeps=10)
    obs = env.reset()
    for _ in range(5):
        actions = [env.action_space[i].sample() for i in range(env.n)]
        obs, rew, done, info = env.step(actions)
        print(f"rew: {rew}")
