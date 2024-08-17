# Libraries
from gymnasium import Env
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
# Local dependencies
from algorithms import RLAgent

# Universal Tester Class
class Tester:
	def __init__(self, env:Env, agent:RLAgent, device: torch.device = torch.device("cpu")) -> None:
		self.env = env
		self.agent = agent
		self.testing_history = {
			"steps": [],
			"reward": [],
		}
		self.device = device

	def get_one_hot_encoding(self, state):
		state_arr = np.zeros(self.env.observation_space.n)
		state_arr[state] = 1
		# Convert NumPy array to PyTorch Tensor
		return torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)

	def record_episode(self, steps:int, reward:float) -> None:
		"""Method to record each test episode"""
		self.testing_history["steps"].append(steps)
		self.testing_history["reward"].append(reward)

	def print_actions(self, actions:list[int]) -> None:
		"""Method to print the actions taken"""
		action_map = {0: "move south",
						1: "move north",
						2: "move east",
						3: "move west",
						4: "pickup passenger",
						5: "drop off passenger"}
		print("Actions taken: [", end="")
		for action in actions:
			if action == 5:
				print(action_map[action], end=", ")
			else:
				print(action_map[action], end=", ")
		print("]")

	def test_episode(self):
		observation, _ = self.env.reset()
		total_reward = 0
		step = 0
		actions = []

		# Iterate steps
		while True:
			# Get the one hot encoding
			processed_observation = self.get_one_hot_encoding(observation)
			# Select best action (test)
			action = self.agent.select_action(processed_observation)
			actions.append(action)
			# Perform the action
			observation, reward, terminated, truncated, _ = self.env.step(action)
			# Increment reward
			total_reward += reward
			if terminated or truncated:
				break
			# Increment step
			step += 1
		# Print the actions taken
		self.print_actions(actions)
		# Return the reward and steps
		return total_reward, step+1

	def average_metrics(self):
		return np.average(self.testing_history["reward"]), np.average(self.testing_history["steps"])

	def stdev_metrics(self):
		return np.std(self.testing_history["reward"]), np.std(self.testing_history["steps"])

	def simulation(self, episodes:int):
		time_start = time.time()
		for episode in range(episodes):
			total_reward, total_steps = self.test_episode()
			self.record_episode(total_steps, total_reward)
			print(f"Episode: {episode + 1}/{episodes}, Steps: {total_steps}, Reward: {int(total_reward)}")

		print("Testing Completed")
		time_end = time.time()
		print(f"Time Taken: {time_end - time_start} seconds")

		avg_reward, avg_steps = self.average_metrics()
		print(f"Average reward: {avg_reward}, Average steps: {avg_steps}")
		std_reward, std_steps = self.stdev_metrics()
		print(f"Standard deviation reward: {std_reward}, Standard deviation steps: {std_steps}")

	def plot_reward_dist(self):
		sns.displot(self.testing_history["reward"])
		plt.title(f'Reward Distribution over {len(self.testing_history["reward"])} Episodes')
		plt.xlabel('Rewards')
		plt.show()