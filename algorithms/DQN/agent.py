# Libraries
import torch
import torch.optim as optim
import torch.nn.functional as F
from gymnasium.spaces import Discrete
import random
import math
from typing import Callable

# Local dependencies
from .network import DQN
from .memory import Transition
from ..base import RLAgent

class DQNAgent(RLAgent):
	def __init__(self, action_space: Discrete, observation_space: Discrete,
				gamma:float = 0.95, alpha:float = 1e-3, tau:float = 0.005,
				eps0:float = 0.9, eps_min:float = 0.05, eps_decay:int = 1000,
				optimizer:optim.Optimizer = optim.Adam, loss:Callable = F.mse_loss,
				device:torch.device = torch.device("cpu")):
		# Init Parent Class
		super().__init__(action_space=action_space, observation_space=observation_space)
		# Set up the device
		self.device = device # Accelerator type
		# Initialize Main Q-Network
		self.main_q_network = DQN(observation_space.n, action_space.n).to(device)
		# Initialize Target Q-Network
		self.target_q_network = DQN(observation_space.n, action_space.n).to(device)
		# Set target_q_network weights to be the same as the main network
		self.target_q_network.load_state_dict(self.main_q_network.state_dict())

		# Setup exploration hyperparameters
		self.gamma = gamma # discount factor
		self.alpha = alpha # learning rate
		self.tau = tau # update rate of target network
		self.eps0 = eps0 # epsilon
		self.epsilon = eps0 # epsilon
		self.eps_min = eps_min #final value of epsilon
		self.eps_decay = eps_decay # rate of exponential decay of epsilon

		# Setup training hyperparameters
		self.optimizer = optimizer(lr=self.alpha, params=self.main_q_network.parameters()) # training optimizer
		self.loss = loss # training loss

	# Mode
	def eval(self):
		self.epsilon = 0

	def train(self):
		self.epsilon = self.eps0

	def load_model(self, path:str, weights_only=False):
		"""Method to load a model"""
		self.main_q_network.load_state_dict(torch.load(path, map_location=self.device, weights_only=weights_only))
		self.target_q_network.load_state_dict(torch.load(path, map_location=self.device, weights_only=weights_only))

	def save_model(self, path:str):
		torch.save(self.main_q_network.state_dict(), path)

	def decay_epsilon(self, episodes:int):
		self.epsilon = self.eps_min + (self.eps0 - self.eps_min) * math.exp(-1 * episodes / self.eps_decay)

	def select_action(self, state):
		"""Method to simulate the agent selecting an action from the action space"""
		sample = random.random()
		# If random sample is more than threshold, we exploit
		if sample > self.epsilon:
			# Exploitation, simply take the Arg Max of the Q(s, a)
			with torch.no_grad():
				return self.main_q_network(state).max(1).indices.view(1, 1).item()
		else:
			# Exploration, simply take a random action
			return torch.tensor([[self.action_space.sample()]], device=self.device, dtype=torch.long).item()

	def compute_loss(self, Q_predicts, Q_targets):
		"""Method to compute loss"""
		# Calculate the Loss
		loss = self.loss(Q_predicts, Q_targets.unsqueeze(1))
		return loss

	def update_target_network(self):
		"""Method to update the target network's weights"""
		# Get the target network state dictionary
		target_net_state_dict = self.target_q_network.state_dict()
		main_net_state_dict = self.main_q_network.state_dict()
		# Update the target network weights
		for key in main_net_state_dict:
			target_net_state_dict[key] = main_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1-self.tau)
		# Load the updated weights back into the target network
		self.target_q_network.load_state_dict(target_net_state_dict)

	def train_step(self, batch: Transition):
		"""Method to simulate each step the agent takes during the training phase"""
		# Unpack transitions
		state_batch = torch.cat(batch.state)
		action_batch = torch.cat(batch.action)
		reward_batch = torch.cat(batch.reward)

		# Create masks for the non-terminating statee
		non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
												batch.next_state)), device=self.device, dtype=torch.bool)
		non_final_next_states = torch.cat([s for s in batch.next_state
													if s is not None])

		# Calculate the Q_predict Q(a,s) with the main network
		Q_predicts = self.main_q_network(state_batch).gather(1, action_batch)

		# In case the state is final, we want to handle it with a 0; thus, we initialize a torch.zeros Tensor.
		next_state_values = torch.zeros(len(batch), device=self.device)
		# Retrieve the Maximum Q(a',s') values
		with torch.no_grad(): # No gradient, since the one we're directly training is only the main network.
			next_state_values[non_final_mask] = self.target_q_network(non_final_next_states).max(1).values

		# Calculate Q_Target = reward + discount_factor * Max Q(a',s')
		Q_targets = reward_batch + (self.gamma * next_state_values)
		#Compute Loss
		loss = self.compute_loss(Q_predicts=Q_predicts, Q_targets=Q_targets)

		# Zero grad
		self.optimizer.zero_grad()
		# Backpropagate
		loss.backward()
		# In-place gradient clipping
		torch.nn.utils.clip_grad_value_(self.main_q_network.parameters(), 100)
		# Update network params
		self.optimizer.step()