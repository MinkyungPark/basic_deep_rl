from pyrsistent import v
import torch

class BitFlipEnvironment:
    def __init__(self, bits):
        self.bits = bits
        self.state = torch.zeros((self.bits, ))
        self.goal = torch.zeros((self.bits, ))
        self.reset()

    def reset(self):
        self.state = torch.randint(2, size=(self.bits, ), dtype=torch.float)
        self.goal = torch.randint(2, size=(self.bits, ), dtype=torch.float)
        if torch.equal(self.state, self.goal):
            self.reset()
        return self.state.clone(), self.goal.clone()

    def step(self, act_num):
        self.state[act_num] = 1 - self.state[act_num]
        reward, done = self.compute_reward(self.state, self.goal)
        return self.state.clone(), reward, done

    def render(self):
        print(f"State: {self.state.tolist()}")
        print(f"Goal: {self.goal.tolist()}")

    @staticmethod
    # to generate new goal
    def compute_reward(state, goal):
        done = torch.equal(state, goal)
        return torch.tensor(0.0 if done else -1.0), done