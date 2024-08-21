from algorithms.PPO import PPOAgent, RewardStepsCallback

class PPOTrainer:
    def __init__(self, agent:PPOAgent, total_timesteps:int, save_frequency:int = 100_000, 
                 save_path:str = './weights', save_checkpoint:bool = False) -> None:
        self.agent = agent
        self.total_timesteps = total_timesteps
        self.save_path = save_path

        self.reward_callback = RewardStepsCallback(save_freq=save_frequency, save_path=self.save_path, save_checkpoint=save_checkpoint)

    def train(self) -> RewardStepsCallback:
        agent = self.agent.get_model()
        agent.learn(total_timesteps=self.total_timesteps, callback=self.reward_callback)
        return self.reward_callback
    
    