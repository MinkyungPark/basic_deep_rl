# HER : https://github.com/AlexHermansson/hindsight-experience-replay/blob/master/bitflip.py 
# https://arxiv.org/pdf/1707.01495.pdf

import random
import torch
from tqdm import tqdm
from collections import namedtuple

from src.test_environment import BitFlipEnvironment
from model.double_dqn import DQNAgent


def bitflip_train(num_bits=10, num_epochs=10, hindsight_replay=True,
          eps_max=0.2, eps_min=0.0, exploration_fraction=0.5,
          stratgy='final', model=DQNAgent):

    Transition = namedtuple("Transition", field_names="state action reward next_state done")
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    future_k = 4
    num_cycles = 50
    num_episodes = 16
    num_opt_steps = 40

    env = BitFlipEnvironment(num_bits)
    num_actions = num_bits
    state_size = 2 * num_bits
    agent = model(state_size, num_actions, device)

    success_rate = 0.0
    success_rates = []

    def store_episode():
        successes = 0
        for _ in range(num_episodes):
            episode_trajectory = []
            state, goal = env.reset()

            # Generate transitions by agent
            for step in range(num_bits): # timesteps
                state_goal = torch.cat((state, goal))
                action = agent.take_action(state_goal, eps)
                next_state, reward, done = env.step(action.item())
                episode_trajectory.append(Transition(state, action, reward, next_state, done))
                state = next_state
                if done:
                    successes += 1
                    break

            # Store transitions
            done_steps = step
            for t in range(done_steps):
                state, action, reward, next_state, done = episode_trajectory[t]
                state_goal, next_state_goal = torch.cat((state, goal)), torch.cat((next_state, goal))
                # store with origin goal
                agent.store_experience(state_goal, action.to(device), reward, next_state_goal, done)

                # HER
                if hindsight_replay:
                    for _ in range(future_k):
                        # final : last state
                        # future : one of current timestep ~ done timestep states
                        goal_idx = random.randint(t, done_steps) if stratgy == 'future' else -2 # else final
                        new_goal = episode_trajectory[goal_idx].next_state 
                        new_reward, new_done = env.compute_reward(next_state, new_goal)
                        state_new_goal = torch.cat((state, new_goal))
                        next_state_goal = torch.cat((next_state, new_goal))
                        # store with additional goal
                        agent.store_experience(state_new_goal, action.to(device), new_reward, next_state_goal, new_done)
        return successes

    for epoch in range(num_epochs):
        # epsilon(Exploration) decay linearly from eps_max to eps_min
        eps = max(eps_max - epoch * (eps_max - eps_min) / int(num_epochs * exploration_fraction), eps_min)
        print(f"Epoch: {epoch + 1}, exploration: {100 * eps:.0f}%, success rate: {success_rate:.2f}")
        
        successes = 0
        for _ in tqdm(range(num_cycles), desc='Cycle'):
            successes += store_episode()
            for _ in range(num_opt_steps):
                agent.optimize_model()
            agent.update_target_network()

        success_rate = successes / (num_episodes * num_cycles)
        success_rates.append(success_rate)

    return success_rates
