from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient


# def make_env(seed):

#     def _init():
        # env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
#         env.seed(seed)
#         return env

#     return _init
def make_env():

    def _init():
        env = HIVPatient(domain_randomization=False)
        env = TimeLimit(env, max_episode_steps=200)
        return env
        
    return _init

class ProjectAgent:

    def __init__(self, num_envs=10):

        # env_fns = [make_env(seed=i) for i in range(num_envs)]
        self.env = SubprocVecEnv([make_env() for _ in range(num_envs)])

        self.model = DQN(
            "MlpPolicy", 
            self.env,
            gradient_steps=8,
            gamma=0.99,
            target_update_interval=1000,
            learning_rate=1e-3, 
            # buffer_size=10000, 
            batch_size=64,
            policy_kwargs=dict(net_arch=[256, 256]),
            tau=0.03,
            verbose=1)

    def act(self, observation, use_random=False):
        action, _states = self.model.predict(observation, deterministic=not use_random)
        return action

    def save(self, path):
        self.model.save(path)

    def load(self):
        self.model = DQN.load("dqn_hiv_model_2", env=self.env)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)
        self.save("/home/onyxia/work/mva-rl-assignment-eliott-esiee/models/dqn_hiv_model_2")

if __name__ == '__main__':
    agent = ProjectAgent()
    agent.train(total_timesteps=500000)
