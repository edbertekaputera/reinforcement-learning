# Optimizing the Taxi Problem Using Reinforcement Learning: A Comparative Analysis of Model-Free Algorithms

This project presents itself as a set of approaches relevant in the reinforcement learning space to tackle the Taxi problem. This repository is the result of the work done under the module CSCI323 - Modern AI from UOW. Authored by Group: Real Life Taxi.


![](https://i.postimg.cc/L4NNf2Kt/image.png)

## Problem Statement
In this environment, the taxi starts on a random square on the grid. The passenger is placed on a random square at the start (red/green/yellow/blue). The taxi must then move to the passenger, perform a pickup action, deliver the passenger to a specified destination (red/green/yellow/blue) and perform a drop-off action

## Problem Objective
The main purpose of this project is to develop an RL agent that can most effectively solve the Taxi problem. This is found by optimizing agent performance and comparing the three RL approaches we implemented. Based on their performances, we can observe the differences in results and time to train each algorithm.

## Potential Applications
- Ride sharing services: Uber and Lyft require efficient routing to pick up and drop off passengers. Estimating the optimal route costs could also help these applications manage dynamic pricing, accounting for traffic and passenger density.
- Parcel delivery: Parcel delivery routes can be optimized by efficiently handling delivery points, much like picking up and dropping off passengers.
- Ambulance dispatch: Efficient routes for ambulances through grid-like urban environments are mission-critical in reducing response times and saving lives.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/edbertekaputera/reinforcement-learning.git
   ```
2. Install Dependencies
   ```sh
   conda create -p ./env python=3.11 -y
   conda activate ./env
   pip install -r requirements.txt
   ```
3. Get Usage Help
    ```sh
   python test.py --help
   ```

## Findings
### Q-Learning
<div style="display: flex; justify-content: space-between;">
  <img src="https://i.postimg.cc/Gt6RzfSd/image.png" alt="Image 1" style="width: 30%;">
  <img src="https://i.postimg.cc/FHyvgDbX/image.png" alt="Image 2" style="width: 30%;">
  <img src="https://i.postimg.cc/gcxnqbx9/image.png" alt="Image 3" style="width: 30%;">
</div>
<br>

![](https://i.postimg.cc/0N5NzLTB/image.png)

### DQN
<div style="display: flex; justify-content: space-between;">
  <img src="https://i.postimg.cc/0jsFQq36/image.png" alt="Image 1" style="width: 30%;">
  <img src="https://i.postimg.cc/QxjvZh1w/image.png" alt="Image 2" style="width: 30%;">
  <img src="https://i.postimg.cc/5NpTxTm0/image.png" alt="Image 3" style="width: 30%;">
</div>
<br>

![](https://i.postimg.cc/LXd0Cpmy/image.png)

### PPO
<div style="display: flex; justify-content: space-between;">
  <img src="https://i.postimg.cc/cLRKg7rr/image.png" alt="Image 1" style="width: 50%;">
  <img src="https://i.postimg.cc/zBB3c9pg/image.png" alt="Image 2" style="width: 50%;">

</div>
<br>

![](https://i.postimg.cc/0N5NzLTB/image.png)

## Conclusion
- Q learning as the base approach directly calculates q tables, and can guarantee convergence on the problem, however it is important to note that utilizing this on a more complex problem would lead to a very computationally and memory heavy solution.
- ML approaches like DQN maybe more optimal because it does not store q tables, rather we are estimating the q value on runtime. In addition, our results showed that its capabilities are very impressive as the performance converged at a very fast rate and it can handle more complex environments without having to scale the computation and memory usage as much.
- In contrast, although PPO shows convergence towards the solution, it also shows relatively more instability. This may be caused by its overcomplexity on this simple problem, hinting that it might be an over-parameterized solution.

## Authors
- [@edbertekaputera](https://www.github.com/edbertekaputera)
- [@edkesuma](https://www.github.com/edkesuma)
- [@JonathanBastineKho](https://www.github.com/jonathanbastinekho)
- [@jovanjoto](https://www.github.com/jovanjoto)
- [@shangji1](https://github.com/shangji1)
- [@stefananeshka](https://www.github.com/stefananeshka)


