from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch.nn as nn


def make_env():

    def _init():
        env = HIVPatient(domain_randomization=False)
        env = TimeLimit(env, max_episode_steps=200)
        return env
        
    return _init


class ProjectAgent:

    def __init__(self, num_envs=20):

        env = SubprocVecEnv([make_env() for _ in range(num_envs)])
        self.env = VecNormalize(
            env, 
            training=True,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0
        )

        self.model = DQN(
            "MlpPolicy", 
            self.env,
            gradient_steps=8,
            gamma=0.95,
            target_update_interval=600,
            learning_rate=1e-3, 
            batch_size=128,
            policy_kwargs={
                "net_arch": [256, 256, 256, 256],  
                "activation_fn": nn.ReLU
            },
            tau=0.02,
            verbose=1)

    def act(self, observation, use_random=False):
        action, _states = self.model.predict(observation, deterministic=not use_random)
        return action

    def save(self, path):
        self.model.save(path)
        self.env.save("/home/onyxia/work/mva-rl-assignment-eliott-esiee/model_7/vecnormalize.pkl")

    def load(self):
        env = SubprocVecEnv([make_env() for _ in range(20)])
        self.env = VecNormalize.load("vecnormalize.pkl", env)
        self.env.training = False
        self.env.norm_reward = False

        self.model = DQN.load("dqn_hiv_model_7", env=self.env)

    def train(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps)
        self.save("/home/onyxia/work/mva-rl-assignment-eliott-esiee/model_7/dqn_hiv_model_7")

if __name__ == '__main__':
    agent = ProjectAgent()
    agent.train(total_timesteps=1000000)
