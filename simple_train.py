from tf2marl.agents import MASACAgent, MAD3PGAgent, MATD3Agent, MADDPGAgent
from tf2marl.multiagent.environment import MultiAgentEnv
from tf2marl.common.util import softmax_to_argmax
from tf2marl.common.simple_logger import SimpleLogger
import time
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="matd3", help="policy for good agents (matd3 or maddpg)")
    parser.add_argument("--adv-policy", type=str, default="matd3", help="policy of adversaries (matd3 or maddpg)")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--update-rate", type=int, default=100, help="after this many steps the critic is trained")
    parser.add_argument("--policy-update-rate", type=int, default=2,
                        help="after this many critic updates the target networks and policy are trained")
    parser.add_argument("--use-critic-noise", action="store_true", default=False,
                        help="use noise in critic update next action")
    parser.add_argument("--use-critic-noise-self", action="store_true", default=False,
                        help="use noise in critic update next action")
    parser.add_argument("--critic-action-noise-stddev", type=float, default=0.2)
    parser.add_argument("--action-noise-clip", type=float, default=0.5)
    parser.add_argument("--critic-zero-if-done", action="store_true", default=False,
                        help="set q value to zero in critic update after done")

    parser.add_argument("--buff-size",default=1e6,type=int, help="size of the replay buffer")
    parser.add_argument("--num-layers", default=2, type=int, help="number of hidden layer per network")
    parser.add_argument("--tau", default=0.02, type=float, help="update rate for target network")
    parser.add_argument("--alpha", default=0.6, type=float, help="alpha value - prioritization vs random")
    parser.add_argument("--beta", default=0.5, type=float, help="beta value controlling importance sampling")
    parser.add_argument("--priori-replay", default=False, action='store_true', help="enable prioritize replay")
    parser.add_argument("--use-target-action", action="store_true", default=True, help="use target action in environment, instead of normal action")
    parser.add_argument("--hard-max", action="store_true", default=True, help="use straight-through gumbel")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='def_exp_name', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--real-q-log", action="store_true", default=False,
                        help="Evaluates approx. real q value after every 5 save-rates")
    parser.add_argument("--q-log-ep-len", type=int, default=200, help="Number of steps per state in q_eval")
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False,
                        help="Saves all locations and termination locations")
    parser.add_argument("--benchmark-iters", type=int, default=10000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    parser.add_argument("--record-episodes", action="store_true", default=False, help="save rgb arrays of episodes")
    return parser.parse_args()


class SimpleExperiment:

    def __init__(self, config, name: str, eid: int):
        self.config = config
        self.name = name
        self._id = eid

    def log_scalar(self, key_name, value, step=-1):
        print(f"{step} : {key_name} -> {value}")


def make_env(scenario_name) -> MultiAgentEnv:
    """
    Create an environment
    :param scenario_name:
    :return:
    """
    if scenario_name == 'inversion':
        from tf2marl.multiagent.scenarios.inversion import Scenario
    elif scenario_name == 'maximizeA2':
        from tf2marl.multiagent.scenarios.maximizeA2 import Scenario
    elif scenario_name == 'simple_adversary':
        from tf2marl.multiagent.scenarios.simple_adversary import Scenario
    elif scenario_name == 'simple_crypto':
        from tf2marl.multiagent.scenarios.simple_crypto import Scenario
    elif scenario_name == 'simple_push':
        from tf2marl.multiagent.scenarios.simple_push import Scenario
    elif scenario_name == 'simple_reference':
        from tf2marl.multiagent.scenarios.simple_reference import Scenario
    elif scenario_name == 'simple_speaker_listener':
        from tf2marl.multiagent.scenarios.simple_speaker_listener import Scenario
    elif scenario_name == 'simple_tag':
        from tf2marl.multiagent.scenarios.simple_tag import Scenario
    elif scenario_name == 'simple_world_comm':
        from tf2marl.multiagent.scenarios.simple_world_comm import Scenario
    else:
        from tf2marl.multiagent.scenarios.simple_spread import Scenario

    scenario = Scenario()
    # create world
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def create_agent(alg_name, index: int, env: MultiAgentEnv, exp):
    conf = exp.config
    if alg_name == 'maddpg':
        ret_agent = MADDPGAgent(env.observation_space, env.action_space,
                                index, conf.batch_size,
                                conf.buff_size,
                                conf.lr, conf.num_layers,
                                conf.num_units, conf.gamma, conf.tau, conf.priori_replay,
                                alpha=conf.alpha,
                                max_step=conf.num_episodes * conf.max_episode_len,
                                initial_beta=conf.beta, _run=exp)
    elif alg_name == 'matd3':
        ret_agent = MATD3Agent(env.observation_space, env.action_space, index, conf.batch_size,
                               conf.buff_size,
                               conf.lr, conf.num_layers,
                               conf.num_units, conf.gamma, conf.tau, conf.priori_replay, alpha=conf.alpha,
                               max_step=conf.num_episodes * conf.max_episode_len, initial_beta=conf.beta,
                               policy_update_freq=conf.policy_update_rate,
                               target_policy_smoothing_eps=conf.critic_action_noise_stddev,_run=exp)
    elif alg_name == 'mad3pg':
        ret_agent = MAD3PGAgent(env.observation_space, env.action_space, index, conf.batch_size,
                                conf.buff_size,
                                conf.lr, conf.num_layers,
                                conf.num_units, conf.gamma, conf.tau, conf.priori_replay, alpha=conf.alpha,
                                max_step=conf.num_episodes * conf.max_episode_len, initial_beta=conf.beta,
                                num_atoms=conf.num_atoms, min_val=conf.min_val, max_val=conf.max_val, _run=exp
                                )
    elif alg_name == 'masac':
        ret_agent = MASACAgent(env.observation_space, env.action_space, index, conf.batch_size,
                               conf.buff_size,
                               conf.lr, conf.num_layers, conf.num_units, conf.gamma, conf.tau, conf.priori_replay,
                               alpha=conf.alpha,
                               max_step=conf.num_episodes * conf.max_episode_len, initial_beta=conf.beta,
                               entropy_coeff=conf.entropy_coeff, policy_update_freq=conf.policy_update_rate,_run=exp)
    else:
        raise RuntimeError(f'Invalid Class - {alg_name} is unknown')

    return ret_agent


def train(conf):
    env = make_env(conf.scenario)
    exp = SimpleExperiment(conf, 'tester', 12)
    logger = SimpleLogger('tester', exp, len(env.agents), env.n_adversaries, conf.save_rate)

    agents = []
    for agent_idx in range(env.n_adversaries):
        agents.append(create_agent(conf.adv_policy, agent_idx, env, exp))
    for agent_idx in range(env.n_adversaries, env.n):
        agents.append(create_agent(conf.good_policy, agent_idx, env, exp))
    print(f'Using good policy {conf.good_policy} and adv policy {conf.adv_policy}')

    # todo: Load previous results, if necessary

    obs_n = env.reset()

    print('Starting iterations...')
    while True:
        # get action
        if conf.use_target_action:  # note: what is target ????????
            # adding an extra axis to the observation
            action_n = [agent.target_action(obs.astype(np.float32)[None])[0] for agent, obs in zip(agents, obs_n)]
        else:
            action_n = [agent.action(obs.astype(np.float32)) for agent, obs in zip(agents, obs_n)]

        # environment step
        if conf.hard_max:
            action_n = softmax_to_argmax(action_n, agents)
        else:
            action_n = [action.numpy() for action in action_n]

        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        logger.episode_step += 1

        done = all(done_n)
        terminal = (logger.episode_step >= conf.max_episode_len)
        done = done or terminal

        # collect experience
        for i, agent in enumerate(agents):
            agent.add_transition(obs_n, action_n, rew_n[i], new_obs_n, done)
        obs_n = new_obs_n

        for ag_idx, rew in enumerate(rew_n):
            logger.cur_episode_reward += rew
            logger.agent_rewards[ag_idx][-1] += rew

        if done:
            obs_n = env.reset()
            episode_step = 0
            logger.record_episode_end(agents)

        logger.train_step += 1

        # policy updates
        train_cond = not conf.display
        for agent in agents:
            if train_cond and len(agent.replay_buffer) > conf.batch_size * conf.max_episode_len:
                if logger.train_step % conf.update_rate == 0:  # only update every 100 steps
                    q_loss, pol_loss = agent.update(agents, logger.train_step)

        # for displaying learned policies
        if conf.display:
            time.sleep(0.1)
            env.render()

        # saves logger outputs to a file similar to the way in the original MADDPG implementation
        if len(logger.episode_rewards) > conf.num_episodes:
            logger.experiment_end()
            return logger.get_sacred_results()


if __name__ == "__main__":
    cmdargs = parse_args()
    train(cmdargs)
