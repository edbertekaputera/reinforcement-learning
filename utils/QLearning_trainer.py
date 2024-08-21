import gymnasium as gym 
import numpy as np 
import torch 
import time 
import math 

from algorithms.QLearning import QLearningAgent  

class QLearningTrainer:
    def __init__(self, env:gym.Env, agent:QLearningAgent, episodes: int, initial_exploration_prob: float, exploration_decay_rate: int, final_exploration_prob: float) -> None:
        self.agent = agent  
        self.env = env
        self.episodes = episodes
        self.initial_exploration_prob = initial_exploration_prob
        self.exploration_decay_rate = exploration_decay_rate
        self.final_exploration_prob = final_exploration_prob
        self.training_history = {
            "steps": [],
            "reward": [],
            "epsilon": []
        }

    def record_episode(self, steps: int, reward: int, epsilon: float) -> None:
        self.training_history["steps"].append(steps)
        self.training_history["reward"].append(reward)
        self.training_history["epsilon"].append(epsilon)
    
    def reset_training_history(self) -> None:
        self.training_history = {
            "steps": [],
            "reward": [],
            "epsilon": []
        }
    
    def get_exploration_prob(self, episode: int) -> float:
        return self.final_exploration_prob + (self.initial_exploration_prob - self.final_exploration_prob) * math.exp(-1 * episode / self.exploration_decay_rate)

    def train_episode(self) -> tuple[int, int, float]:
        total_reward: int = 0
        steps: int = 0
        current_state, _ = self.env.reset()
        terminated: bool = False
        truncated: bool = False

        exploration_prob = self.get_exploration_prob(steps)
        
        while not terminated and not truncated:
            # Select an action using the agent's select_action method
            action = self.agent.select_action(current_state, exploration_prob)
            
            # Perform the action and observe the outcome
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            # Update the Q-table using the agent's update_q_table method
            self.agent.update_q_table(current_state, action, reward, next_state)
            
            total_reward += reward
            steps += 1
            current_state = next_state
        
        return total_reward, steps, exploration_prob

    def train(self) -> None:
      for episode in range(self.episodes):
          total_reward, steps, exploration_prob = self.train_episode()
          self.record_episode(steps, total_reward, exploration_prob)
          if (episode + 1) % 1000 == 0:
              print(f"Episode {episode + 1} completed")

