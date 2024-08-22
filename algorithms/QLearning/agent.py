import numpy as np
from gymnasium.spaces import Space
import random
import math

from ..base import RLAgent


class QLearningAgent(RLAgent):
	def __init__(self, action_space: Space, observation_space: Space, 
				eps0:float = 0.9, eps_min:float = 0.05, eps_decay:int = 1000, 
				learning_rate: float = 1e-3, discount_factor: float = 0.95) -> None:
		super().__init__(action_space, observation_space)
		self.learning_rate = learning_rate
		self.discount_factor = discount_factor
		self.Q_table = np.zeros((observation_space.n, action_space.n))
		self.eps0 = eps0 # epsilon
		self.epsilon = eps0 # epsilon
		self.eps_min = eps_min #final value of epsilon
		self.eps_decay = eps_decay # rate of exponential decay of epsilon

	def select_action(self, state: int) -> int:
		# Select an action using epsilon-greedy strategy
		sample = random.random()
		if sample < self.epsilon:
			return self.action_space.sample()  # Exploration
		else:
			return np.argmax(self.Q_table[state])  # Exploitation

	def decay_epsilon(self, episodes:int):
		self.epsilon = self.eps_min + (self.eps0 - self.eps_min) * math.exp(-1 * episodes / self.eps_decay)

	# Mode
	def set_exploit(self):
		self.epsilon = 0

	def set_explore(self):
		self.epsilon = self.eps0

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