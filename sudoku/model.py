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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size: list, output_size):
        super().__init__()
        self.input_size = input_size
        self.layers = []
        hidden_size = [input_size] + hidden_size + [output_size]
        for i in range(len(hidden_size) - 1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
            self.layers.append(nn.ReLU())

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.view(-1, self.input_size)

        for layer in self.layers:
            x = layer(x)
        return x # x corresponds to Q values for each action

    def save(self, folder_path='.', file_name='model.pth'):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_name = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma, device):
        self.lr = lr
        self.gamma = gamma
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.losses = []
        self.mean_losses = []
        self.device = device

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)

        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        state = state.view(-1, self.model.input_size)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx])) 

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward() 

        self.losses.append(loss.item())
        self.mean_losses.append(sum(self.losses) / len(self.losses))

        self.optimizer.step()


if __name__ == '__main__':
    model = Linear_QNet(81 * 2, [16384, ], 810)
    count = 0 
    for param in model.parameters():
        count += 1
        print(param)

    print(len(list(model.layers[0].parameters())))
    print(count) 
    print(model)