from gymnasium import Space
from torch import Tensor
import torch.nn as nn
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO

from ..base import RLAgent

class PPOAgent(RLAgent):
    def __init__(self, action_space: Space, observation_space: Space,
                n_environment:int,
                lr:float = 0.005, mini_batch:int = 128, gamma:float = 0.95) -> None:
        super().__init__(action_space, observation_space)
        self.lr = lr
        self.mini_batch = mini_batch
        self.gamma = gamma

        self.vec_env = make_vec_env("Taxi-v3", n_envs=n_environment)

        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            policy_kwargs=dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],
                activation_fn=nn.Tanh
            ),
            learning_rate=self.lr,
            batch_size=self.mini_batch,
            gamma=self.gamma,
            verbose=1
        )

    def select_action(self, state) -> int | float | Tensor:
        action, _states = self.model.predict(state, deterministic=True)
        return action.item()

    def save_model(self, path:str) -> None:
        self.model.save(path=path)

    def load_model(self, path:str) -> None:
        self.model = PPO.load(path=path)

    def get_model(self) -> PPO:
        return self.model