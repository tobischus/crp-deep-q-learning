# ---------------------------------------------------------------------------------------
# Title: Container Relocation Problem: Application of a Machine Learning Approach
# Author: Tobias Schuster
# Student ID: 2827647
# Email: tobias.schuster@stud.tu-darmstadt.de
# Submission Date: 21.08.2024
# Institution: Technical University of Darmstadt, Law and Economics Department, Management Science/Operations Research

# Supervisor: Prof. Dr. Felix Weidinger
#
# Description:
# This script implements a Deep Q-Learning algorithm to solve the Container Relocation Problem (CRP).
# The goal of the algorithm is to minimize the number of container relocations required in a container yard.
# 
# Technologies Used:
# - Python 3.9.13
# - PyTorch 2.0.0+cpu
# - NumPy 1.25.0
# ---------------------------------------------------------------------------------------



import time
start_time = time.time()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import os
import glob
import re



# State Representation:
# The state is represented as a list of stacks, where each stack is a list of containers.
class Environment:
    
    def __init__(self, current_state, target_container, max_stack_height):
        self.current_state = current_state
        self.target_container = target_container
        self.max_stack_height = max_stack_height

    # Action Space:
    # Actions are defined as a tuple (from_stack, to_stack), indicating that a container
    # is moved from one stack to another. Only containers blocking the target container can be moved.
    def get_possible_actions(self):
        num_stacks = len(self.current_state)

        source_stacks = [i for i in range(num_stacks) if self.target_container in self.current_state[i]]
        available_stacks = [i for i in range(num_stacks) if len(self.current_state[i]) < self.max_stack_height]
        
        actions = [(i, j) for i in source_stacks for j in available_stacks if i != j]
        return actions


    def update_target_container(self):
        self.target_container += 1
        return self.target_container
    

    # Reward Function:
    # This function gives the agent a negative reward for each relocation (e.g., -2),
    # and positive rewards when a target container is retrieved.
    def step(self, action):
        new_state = Environment([stack[:] for stack in self.current_state], self.target_container, self.max_stack_height)
        from_stack, to_stack = action

        container = new_state.current_state[from_stack].pop()
        new_state.current_state[to_stack].append(container)

        new_state.remove_target_containers()

        done = all(not stack for stack in new_state.current_state)

        reward = -2 + self.count_removed_containers(new_state)
        return new_state, reward, done


    def count_removed_containers(self, next_state):
        initial_container_count = sum(len(stack) for stack in self.current_state)
        next_container_count = sum(len(stack) for stack in next_state.current_state)
        return (initial_container_count - next_container_count)
    

    def remove_target_containers(self):
        found = True
        while found:
            found = False
            for stack in self.current_state:
                if stack and stack[-1] == self.target_container:
                    stack.pop()
                    self.update_target_container()
                    found = True


# Neural Network Design:
# The Q-network consists of an input layer, two hidden layers with ReLU activations,
# and an output layer that predicts the Q-values for each possible action.
class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# DQNAgent:
# Implements a Deep Q-Learning agent for the Container Relocation Problem.
# Uses a policy network to estimate Q-values and a target network to stabilize training.
# Includes epsilon-greedy action selection, experience replay, and periodic target network updates.
class DQNAgent:

    def __init__(self, state_size, action_size, gamma, epsilon, epsilon_decay, epsilon_min, batch_size, learning_rate, replay_freq, target_update_freq):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.hidden_size = 32
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.replay_freq = replay_freq
        self.target_update_freq = target_update_freq
        self.random = 0
        self.not_random = 0

        self.policy_net = DQN(state_size, action_size, self.hidden_size)
        self.target_net = DQN(state_size, action_size, self.hidden_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    # Select an action based on the epsilon-greedy policy:
    # With probability epsilon, a random action is chosen (exploration). Otherwise, the action with the highest 
    # predicted Q-value (among the possible actions) is selected (exploitation). 
    def act(self, state, possible_actions):
        if random.random() < self.epsilon:
            self.random += 1
            return random.choice(possible_actions)
        self.not_random += 1
        max_len = initial_state.max_stack_height

        state = pad_state(state.current_state, max_len)
        state = torch.FloatTensor(state).flatten().unsqueeze(0)  

        with torch.no_grad():
            action_values = self.policy_net(state)
        
        action_indices = [action[1] for action in possible_actions]
        action_values_filtered = action_values[0, action_indices]
        
        best_action_index = torch.argmax(action_values_filtered).item()
        best_action = possible_actions[best_action_index]

        return best_action


    # Experience Replay:
    # This method samples random mini-batches from stored experiences to train the model, reducing data correlation 
    # and stabilizing learning. States are padded to a consistent size before processing.
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        max_len = initial_state.max_stack_height

        states = torch.tensor([pad_state(state, max_len) for state in states], dtype=torch.float32)
        next_states = torch.tensor([pad_state(state, max_len) for state in next_states], dtype=torch.float32)

        batch_size = states.size(0)
        states = states.view(batch_size, -1)
        next_states = next_states.view(batch_size, -1)

        actions = torch.tensor([action[1] for action in actions], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values = self.policy_net(states)
        current_q_values = q_values.gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))


        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    # Training Loop:
    # This loop runs through multiple episodes where the agent interacts with the environment, 
    # selects actions, receives rewards, and updates the Q-values through experience replay. 
    # Periodically, the target network is updated to stabilize training.
    def train(self, num_episodes):
        episode_results = []
        for episode in range(num_episodes):
            steps = 0
            state = initial_state
            state.remove_target_containers()
            done = all(not stack for stack in state.current_state)
            while not done:
                # Select an action using the epsilon-greedy strategy
                possible_actions = state.get_possible_actions()
                action = agent.act(state, possible_actions)
                steps += 1 

                # Execute the action and observe the next state and reward
                next_state, reward, done = state.step(action)

                # Store the experience in memory
                agent.remember(state.current_state, action, reward, next_state.current_state, done)

                # Move to the next state
                state = next_state

            # Perform experience replay and update the model
            if episode % self.replay_freq == 0:
                agent.replay()

            # Periodically update the target network
            if episode % self.target_update_freq == 0:    
                agent.update_target_network()

            agent.update_epsilon()
            episode_results.append((i, steps))
        print(f"Random: {self.random}")   
        print(f"Not-Random: {self.not_random}")  
        return episode_results


def parse_dataset(file_content):

    lines = file_content.strip().split('\n')

    first_line = lines[0].split()
    num_stacks = int(first_line[0])
    num_blocks = int(first_line[1])
    
    container_stacks = []
    
    for line in lines[1:]:
        stack_info = line.split()
        stack = [int(block) for block in stack_info[1:]]
        container_stacks.append(stack)
    
    return num_stacks, num_blocks, container_stacks

def read_all_dat_files(directory):
    datasets = []
    filepaths = glob.glob(os.path.join(directory, "*.dat"))

    def natural_key(filepath):
        filename = os.path.basename(filepath)
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

    filepaths.sort(key=natural_key)
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            file_content = file.read()
            datasets.append(parse_dataset(file_content))
    return datasets


directory = 'dataset'
all_datasets = read_all_dat_files(directory)

def get_caserta_layout(num_tiers, num_stacks):
    data = []
    for dataset in all_datasets:
        if dataset[0] == num_stacks and all(len(stack) == num_tiers for stack in dataset[2]):  
            if dataset not in data:
                data.append(dataset)
    return data


three_by_three_data = get_caserta_layout(3,3)
three_by_four_data = get_caserta_layout(3,4)
three_by_five_data = get_caserta_layout(3,5)
three_by_six_data = get_caserta_layout(3,6)
three_by_seven_data = get_caserta_layout(3,7)
three_by_eight_data = get_caserta_layout(3,8)
four_by_four_data = get_caserta_layout(4,4)
four_by_five_data = get_caserta_layout(4,5)
four_by_six_data = get_caserta_layout(4,6)
four_by_seven_data = get_caserta_layout(4,7)
five_by_four_data = get_caserta_layout(5,4)
five_by_five_data = get_caserta_layout(5,5)
five_by_six_data = get_caserta_layout(5,6)
five_by_seven_data = get_caserta_layout(5,7)
five_by_eight_data = get_caserta_layout(5,8)
five_by_nine_data = get_caserta_layout(5,9)
five_by_ten_data = get_caserta_layout(5,10)
six_by_six_data = get_caserta_layout(6,6)
six_by_ten_data = get_caserta_layout(6,10)
ten_by_six_data = get_caserta_layout(10,6)
ten_by_ten_data = get_caserta_layout(10,10)


def max_stack_height(container_stacks):
    max_height = 0
    for stack in container_stacks:
        stack_height = len(stack)
        if stack_height > max_height:
            max_height = stack_height
    return max_height
def pad_state(state, max_len):
            return [stack + [0] * (max_len - len(stack)) for stack in state]


results = []
shortest_results = []
i=0

# Iterate over different layout sizes, currently set to 3x3 :
# For each layout, initialize the environment and a DQN agent, then train the agent for a number of episodes.
for num_stacks, num_blocks, instance in three_by_three_data:
    print(f"Instanz-Nummer: {i+1} ")
    print(instance)

    # Initialize environment and DQN agent
    initial_state = Environment(instance, target_container=1, max_stack_height=max_stack_height(instance) + 2)
    agent = DQNAgent(state_size=len(instance) * (max_stack_height(instance) + 2),
                 action_size=len(list(itertools.permutations(range(len(instance)), 2))), 
                 gamma=1,
                 epsilon=1.0,
                 epsilon_decay=0.99975,
                 epsilon_min=0.01,
                 batch_size=32,
                 learning_rate=0.001,
                 replay_freq=10,
                 target_update_freq=20)
    
    # Train agent and store results
    result = agent.train(num_episodes=10000)
    fewest_steps_for_instance = min(result, key=lambda x: x[1])
    results.append(result)
    shortest_results.append(fewest_steps_for_instance)
    i += 1



# Results and scatter plots

steps_list = []
instance_size = len(results[0])  

for instance in results:
    steps_instance = [result[1] for result in instance]
    steps_list.append(steps_instance)

steps_array = np.array(steps_list)
steps = np.mean(steps_array, axis=0)

index = list(range(instance_size))
shortest_relocations = [result[1] for result in shortest_results]
shortest_episodes = [result[0] for result in shortest_results]

data_relocation_avg = round(sum(shortest_relocations) / len(shortest_relocations), 3)
data_relocation_rate = round(sum(shortest_relocations) / (len(shortest_relocations) * num_blocks), 3)

runtime = time.time() - start_time

print(f"Anzahl an Relocations: {data_relocation_avg}")
print(f"Relocation rate: {data_relocation_rate}")
print(f"Runtime: {runtime:.4f} seconds")
data = pd.DataFrame({'Episode': index, 'Total Steps': steps})
data.to_csv('episode_rewards.csv', index=False)
plt.figure(figsize=(10, 6))
plt.scatter(index, steps, label='Steps', color='blue', s=10)  
z = np.polyfit(index, steps, 3) 
p = np.poly1d(z)
plt.plot(index, p(index), color='red', label='Trendlinie')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps pro Episode')
plt.legend()
plt.grid(True)
plt.show()