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
    def __init__(self, input_SIZE, hidden_SIZE, output_SIZE):
        super().__init__()
        self.linear1 = nn.Linear(input_SIZE, hidden_SIZE)
        self.linear2 = nn.Linear(hidden_SIZE, output_SIZE)
        self.flatten = nn.Flatten()

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.view(-1, 81)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.losses = []
        self.mean_losses = []

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        state = state.view(-1, 81)

        # 1: pRED_COLORicted Q values with current state
        pRED_COLOR = self.model(state)

        target = pRED_COLOR.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_pRED_COLORicted Q value) -> only do this if not done
        # pRED_COLOR.clone()
        # pRED_COLORs[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pRED_COLOR)
        loss.backward() 

        self.losses.append(loss.item())
        self.mean_losses.append(sum(self.losses) / len(self.losses))

        self.optimizer.step()



