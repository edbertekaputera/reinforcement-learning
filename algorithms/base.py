# abstract base class
from abc import ABC, abstractmethod
from gymnasium.spaces import Space
from torch import Tensor

class RLAgent(ABC):
	def __init__(self, action_space: Space, observation_space: Space) -> None:
		# Initialize action space
		self.action_space = action_space
		# Initialize observation space
		self.observation_space = observation_space
		
	@abstractmethod
	def select_action(self, state) -> int | float | Tensor:
		pass

	@abstractmethod
	def save_model(self, path:str) -> None:
		pass

	def load_model(self, path:str) -> None:
		pass