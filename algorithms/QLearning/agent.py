import torch
import numpy as np
from gymnasium.spaces import Space
from abc import ABC
import random
import math
from typing import Callable

from ..base import RLAgent


class QLearningAgent(RLAgent):
    def __init__(self, action_space: Space, observation_space: Space, learning_rate: float, discount_factor: float) -> None:
        super().__init__(action_space, observation_space)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.Q_table = np.zeros((observation_space.n, action_space.n))

    def select_action(self, state: int, exploration_prob: float) -> int:
        # Select an action using epsilon-greedy strategy
        if np.random.rand() < exploration_prob:
            return self.action_space.sample()  # Exploration
        else:
            return np.argmax(self.Q_table[state])  # Exploitation

    def update_q_table(self, state: int, action: int, reward: float, next_state: int) -> None:
        # Update the Q-table using the Q-learning update rule
        best_future_q = np.max(self.Q_table[next_state])
        self.Q_table[state, action] += self.learning_rate * (reward + self.discount_factor * best_future_q - self.Q_table[state, action])

    def save_model(self, path: str) -> None:
        # Save the Q-table to a file
        np.save(path, self.Q_table)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        # Load the Q-table from a file
        self.Q_table = np.load(path)
        print(f"Model loaded from {path}")

    def get_policy(self) -> np.ndarray:
        # Return the learned policy (the best action for each state)
        return np.argmax(self.Q_table, axis=1)