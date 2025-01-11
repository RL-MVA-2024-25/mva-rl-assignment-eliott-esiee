from stable_baselines3 import DQN
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient


class ProjectAgent:

    def __init__(self):
        self.env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
        self.model = DQN("MlpPolicy", self.env, learning_rate=1e-3, buffer_size=10000, batch_size=64, verbose=1)

    def act(self, observation, use_random=False):
        action, _states = self.model.predict(observation, deterministic=not use_random)
        return action

    def save(self, path):
        self.model.save(path)

    def load(self):
        self.model = DQN.load("/home/onyxia/work/mva-rl-assignment-eliott-esiee/models/dqn_hiv_model", env=self.env)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)
        self.save("/home/onyxia/work/mva-rl-assignment-eliott-esiee/models/dqn_hiv_model")


agent = ProjectAgent()
agent.train(total_timesteps=100000)
