import os
from stable_baselines3.common.callbacks import BaseCallback

class RewardStepsCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, save_checkpoint:bool = False, verbose=0):
        super(RewardStepsCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_checkpoint = save_checkpoint
        self.episode_rewards = []
        self.episode_steps = []

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])
                self.episode_steps.append(info['episode']['l'])
        
        if self.save_checkpoint and self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f'PPO_checkpoint/model_{self.num_timesteps}')
            self.model.save(model_path)
            print(f"Saved model to {model_path}")
        
        return True