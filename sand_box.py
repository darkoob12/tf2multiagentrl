from tf2marl.multiagent.environment import MultiAgentEnv
from tf2marl.multiagent.multi_discrete import MultiDiscrete
from tf2marl.multiagent.scenarios.simple_spread import Scenario
import time

scenario = Scenario()
# create world
world = scenario.make_world()
env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)


def sample_mul(action_space):
    if type(action_space) == MultiDiscrete:
        smaple = action_space.sample()
        ret =[]
        for i, s in enumerate(smaple):
            tmp = [0] * (action_space.high[i] + 1)
            tmp[s] = 1
            ret.append(tmp)
        return ret
    ret = [0] * action_space.n
    ret[action_space.sample()] = 1
    return ret


for i_episode in range(20):
    print(f"starting episode {i_episode + 1}")
    obsv = env.reset()
    for i_step in range(1000):
        env.render('human')
        act = [sample_mul(s) for s in env.action_space]

        obsv, reward, done, info = env.step(act)

        # if done:
        #     print(f"break on {i_step} - {reward}")
        #     break

    print(f"finish after step {i_step} - {reward}")
