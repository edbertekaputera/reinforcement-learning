# Import Libraries
import gymnasium as gym
import numpy as np
import torch
import time

# Local dependency
from algorithms.DQN import DQNAgent, ReplayMemory, Transition

# Trainer
class DQNTrainer:
	def __init__(self, env:gym.Env, agent:DQNAgent, memory: ReplayMemory, save_path="./weights", save_rate=500, device: torch.device = torch.device("cpu")) -> None:
		self.env = env
		self.agent = agent
		# Initialize Replay Memory
		self.memory = memory
		self.training_history = {
			"steps": [],
			"reward": [],
			"epsilon": []
		}
		self.save_path = save_path
		self.save_rate = save_rate
		self.device = device

	def record_episode(self, steps:int, reward:int, epsilon: float) -> None:
		"""Method to record each training episode"""
		self.training_history["steps"].append(steps)
		self.training_history["reward"].append(reward)
		self.training_history["epsilon"].append(epsilon)


	def reset_training_history(self) -> None:
		"""Method to reset the training history"""
		self.training_history = {
			"steps": [],
			"reward": [],
			"epsilon": []
		}

	def get_one_hot_encoding(self, state):
		state_arr = np.zeros(self.env.observation_space.n)
		state_arr[state] = 1
		return state_arr

	def train_episode(self, num_steps_per_update: int = 4, batch_size: int = 128) -> tuple[int, int]:
		"""Method to simulate each training episode"""
		# Reset the environment
		state, _ = self.env.reset()
		state = self.get_one_hot_encoding(state)
		state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
		total_reward:int = 0
		step:int = 0
		# Iterate Steps
		while True:
			# Select an action
			action = self.agent.select_action(state)
			# Perform the action
			next_state, reward, terminated, truncated, _ = self.env.step(action.item())
			next_state = self.get_one_hot_encoding(next_state)
			# Convert to Tensor
			reward = torch.tensor([reward], device=self.device)
			# Increment Reward
			total_reward += reward

			# Check if terminated
			if terminated:
				next_state = None
			else:
				next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)

			# Store the transition in memory
			self.agent.memory.push(state, action, reward, next_state)

			# Move to the next state
			state = next_state

			# Update the target network, copying all weights and biases in DQN
			if step % num_steps_per_update == 0 and len(self.agent.memory) > batch_size:
				# Sample a batch from the replay memory
				transitions = self.memory.sample(batch_size)
				batch = Transition(*zip(*transitions))
				# Perform one step of the optimization (on the main network)
				self.agent.train_step(batch=batch)
				# Update the target network, copying all weights and biases in DQN
				self.agent.update_target_network()
			# Check if the episode is done
			if terminated or truncated:
				break
			# Increment Step
			step += 1
			# Return the reward and steps
		return total_reward, step + 1

	def train(self, num_episodes:int = 5000, num_steps_per_update: int = 4, batch_size: int = 128) -> dict[str, list[int]]:
		"""Method to simulate training"""
		# Reset the training history
		self.reset_training_history()
		time_start = time.time()

		# Iterate Episodes
		for episode in range(num_episodes):
			# Save the model
			if episode % self.save_rate == 0:
				self.agent.save_model(f"{self.save_path}/taxi_model_{episode}.pt")
			# Simulate one training episode
			total_reward, total_steps = self.train_episode(num_steps_per_update, batch_size)
			# Decay Epsilon
			self.agent.decay_epsilon(episode)
			# Log the progress
			self.record_episode(total_steps, total_reward, self.agent.epsilon)
			print(f"Episode: {episode + 1}/{num_episodes}, Steps: {total_steps}, Reward: {int(total_reward)}, Epsilon: {self.agent.epsilon}")

			print("Training Completed")
			time_end = time.time()
			print(f"Time Taken: {time_end - time_start} seconds")
			return self.training_history