import gymnasium as gym 

from algorithms.QLearning import QLearningAgent  

class QLearningTrainer:
	def __init__(self, env:gym.Env, agent:QLearningAgent, save_path="./weights", save_rate=500) -> None:
		self.agent = agent  
		self.env = env
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
		
	def train_episode(self, episode: int) -> tuple[int, int, float]:
		total_reward: int = 0
		steps: int = 0
		current_state, _ = self.env.reset()
		terminated: bool = False
		truncated: bool = False
		# Decay epsilon
		self.agent.decay_epsilon(episode)

		while not terminated and not truncated:
			# Select an action using the agent's select_action method
			action = self.agent.select_action(current_state)
			
			# Perform the action and observe the outcome
			next_state, reward, terminated, truncated, _ = self.env.step(action)
			
			# Update the Q-table using the agent's update_q_table method
			self.agent.update_q_table(current_state, action, reward, next_state)
			
			total_reward += reward
			steps += 1
			current_state = next_state
		
		return total_reward, steps

	def train(self, num_episodes:int = 5000) -> None:
		for episode in range(num_episodes):
			total_reward, steps = self.train_episode(episode)
			self.record_episode(steps, total_reward, self.agent.epsilon)
			if (episode + 1) % 1000 == 0:
				print(f"Episode {episode + 1} completed")
			# Save model
			if episode % self.save_rate == 0:
				self.agent.save_model(f"{self.save_path}/taxi_qtable_{episode}.npy")

