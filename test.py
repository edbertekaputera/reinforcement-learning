# Libraries
import typer
from typing import Annotated, List, Optional
from enum import Enum
from pathlib import Path
from gymnasium import make
import torch
# Local dependencies
from algorithms.DQN import DQNAgent
from algorithms.PPO import PPOAgent
from algorithms.QLearning import QLearningAgent
from utils import Tester

# Enums
class GymEnvironment(str, Enum):
    taxi = "Taxi-v3"

class Algorithm(str, Enum):
	dqn = "dqn"
	ppo = "ppo"
	qlearning = "qlearning"

class AcceleratorType(str, Enum):
	cpu = "cpu"
	mps = "mps"
	cuda = "cuda"

def main(
	model_path: Annotated[Path, typer.Argument(help="Path to model files.")],
	algorithm: Annotated[Algorithm,  typer.Option(help="RL Algorithm")] = Algorithm.dqn,
	environment: Annotated[GymEnvironment, typer.Option(help="Gym Environment ID")] = GymEnvironment.taxi,
	episodes: Annotated[int, typer.Option(min=1, help="Number of episodes")] = 400,
	plot_distribution: Annotated[bool, typer.Option(help="Plot Distribution.")] = False,
	device: Annotated[Optional[AcceleratorType], typer.Option(help="Accelerator Device Type")] = None,
):
	# Check file parameters
	if not model_path.is_file():
		print(f"Specified model file path {model_path} does not exist.")
		raise typer.Abort()
	
	# Accelerator
	if device == AcceleratorType.cuda:
		torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	elif device == AcceleratorType.mps:
		torch_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	if not device:
		# if GPU is to be used
		torch_device = torch.device(
			"cuda" if torch.cuda.is_available() else
			"mps" if torch.backends.mps.is_available() else
			"cpu"
		)
	
	# Setup Environment
	env = make(environment)
	env.reset()

	# Setup Agent
	if algorithm == Algorithm.dqn:
		agent = DQNAgent(
			action_space=env.action_space, 
			observation_space=env.observation_space,
			device=torch_device
		)
		agent.load_model(model_path, weights_only=True)
		agent.eval()
	
	elif algorithm == Algorithm.ppo:
		agent = PPOAgent(
			action_space=env.action_space,
			observation_space=env.observation_space
		)
		agent.load_model(model_path)

	else:
		agent = QLearningAgent(
			action_space=env.action_space,
			observation_space=env.observation_space
		)
		agent.load_model(model_path)
		agent.set_exploit()

	# Setup Tester
	tester = Tester(env=env, agent=agent, device=torch_device)
	# Simulate
	tester.simulation(episodes=episodes)
	# Plot Distribution
	if plot_distribution:
		tester.plot_reward_dist()

if __name__ == "__main__":
	typer.run(main)