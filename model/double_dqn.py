import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from replay import ReplayMemory

class DuelingMLP(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        self.linear = nn.Linear(state_size, 256)
        self.value_head = nn.Linear(256, 1)
        self.advantage_head = nn.Linear(256, num_actions)

    def forward(self, x):
        x = x.unsqueeze(0) if len(x.size()) == 1 else x
        x = F.relu(self.linear(x))
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        action_values = (value + (advantage - advantage.mean(dim=1, keepdim=True))).squeeze()
        return action_values


class DQNAgent:
    def __init__(self, state_size, num_actions, device):
        self.Transition = namedtuple("Experience", field_names="state action reward next_state done")
        self.device = device
        self.state_size = state_size
        self.num_actions = num_actions
        self.gamma = 0.98
        self.batch_size = 128
        self.train_start = 1000
        self.memory = ReplayMemory(int(1e6))

        self.Q_network = DuelingMLP(state_size, num_actions).to(self.device)
        self.target_network = DuelingMLP(state_size, num_actions).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.Q_network.parameters(), lr=0.001)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.push(self.Transition(state, action, reward, next_state, done))

    def update_target_network(self):
        self.target_network.load_state_dict(self.Q_network.state_dict())

    def take_action(self, state, epsilon):
        if random.random() > epsilon:
            return self.greedy_action(state)
        else:
            return torch.randint(self.num_actions, size=())

    def greedy_action(self, state):
        with torch.no_grad():
            return self.Q_network(state.to(self.device)).argmax()

    def optimize_model(self):
        if len(self.memory) < self.train_start:
            return

        experiences = self.memory.sample(self.batch_size)
        batch = self.Transition(*zip(*experiences))

        state_batch = torch.stack(batch.state).to(self.device)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.stack(batch.reward).to(self.device)
        non_final_mask = ~torch.tensor(batch.done)
        non_final_next_states = torch.stack([s for done, s in zip(batch.done, batch.next_state) if not done])
        non_final_next_states = non_final_next_states.to(self.device)

        Q_values = self.Q_network(state_batch)[range(self.batch_size), action_batch]

        next_state_values = torch.zeros(self.batch_size).to(self.device)
        number_of_non_final = sum(non_final_mask)
        with torch.no_grad():
            argmax_actions = self.Q_network(non_final_next_states).argmax(1)
            next_state_values[non_final_mask] = self.target_network(non_final_next_states)[
                range(number_of_non_final), argmax_actions]

        Q_targets = reward_batch + self.gamma * next_state_values

        assert Q_values.shape == Q_targets.shape

        self.optimizer.zero_grad()
        loss = F.mse_loss(Q_values, Q_targets)
        loss.backward()
        self.optimizer.step()