# Work in progress implementation of a sudoku game solver using RL.
# Copyright (C) 2024 Cristian Cerasuolo

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

import torch
import random
import numpy as np
from collections import deque
from game import Game
from model import Linear_QNet, QTrainer
from constants import *
from helper import *
import argparse

class Agent:

    def __init__(self, device):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.device = device
        self.gamma = DISCOUNT_RATE # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(81 * 2, [4096, 8192, 4096], 810).to(torch.device(self.device))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device = torch.device(self.device)) 

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 65 # - self.n_games*0.1
        final_move = (0,0,0)
        if random.randint(0, 200) < self.epsilon:
            # TODO: WE CAN'T MODIFY THE INPUT!!! Actually consider as a violation
            x = random.randint(0, 8)
            y = random.randint(0, 8)
            num = random.randint(0, 9)
            final_move = (x, y, num)
        else:
            state0 = torch.tensor(state, dtype=torch.float).view(1, -1).to(torch.device(self.device))
            prediction = self.model(state0) # prediction is an 810 tensor
            prediction = prediction.reshape(9,9,10)
            final_move = np.unravel_index(np.argmax(prediction.detach().cpu()), prediction.shape)

        return final_move
    

def calculate_completeness(state):
    original_board = state[:9][:]
    input_cells = len(list(filter(lambda x: x != 0, [cell for row in original_board for cell in row])))
    necessary_cells_to_fill = 81 - input_cells
    final_board = state[9:][:]
    total_cells = len(list(filter(lambda x: x != 0, [cell for row in final_board for cell in row])))
    return (total_cells - input_cells) / necessary_cells_to_fill


def train(device, save_period):
    # plot_scores = []
    # plot_mean_scores = []
    # total_score = 0
    # record = 0
    

    violations = 0
    dumb_moves = 0
    total_reward = 0
    change = 0
    games_result = []
    games_violation = []
    games_dumb_moves = []
    games_rewards = []
    games_rewards_mean = []
    games_value_change = []
    games_completeness = []
    games_mean_loss = []

    agent = Agent(device)
    game = Game()
    while True:
        # get old state
        state_old = game.get_state()

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, game_over = game.play_step(final_move)
        if reward[1] == "Violation":
            violations += 1
        elif reward[1] == "Dumb move":
            dumb_moves += 1
        elif reward[1] == "Change":
            change += 1
        total_reward += reward[0]
        
        state_new = game.get_state()


        # train short memory
        agent.train_short_memory(state_old, final_move, reward[0], state_new, game_over)

        # remember
        agent.remember(state_old, final_move, reward[0], state_new, game_over)

        if game_over[0]:
            # train long memory, plot result
            last_state = game.get_state()

            game = Game()
            agent.n_games += 1
            agent.train_long_memory()

            if agent.n_games % save_period == 0:
                agent.model.save(folder_path=os.path.join(SAVE_DIR, "weights"), file_name=f'model_{agent.n_games}.pth')

            print('Game', agent.n_games)
            games_result.append(game_over[1])
            games_violation.append(violations)
            games_dumb_moves.append(dumb_moves)
            games_rewards.append(total_reward)
            games_value_change.append(change)
            games_rewards_mean.append(np.mean(games_rewards))
            games_completeness.append(calculate_completeness(last_state))
            games_mean_loss.append(agent.trainer.get_mean_loss())

            total_reward = 0
            violations = 0
            dumb_moves = 0
            change = 0

            plot_mean(games_rewards, games_rewards_mean, "Rewards", True)
            plots([games_result, games_violation, games_dumb_moves, games_rewards, games_value_change, games_completeness, games_mean_loss], ['Results', 'Violations', 'Dumb moves', 'Rewards', 'Changes', "Completeness", 'Loss'], save = True)

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--save_period", type=int, default=100, help="Period of episodes to save the model")
    args = parser.parse_args()


    train(args.device, args.save_period)
