import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from RobotModel import RobotModel
import sys


class KNData(Dataset):
    def __init__(self, dtype, robot_path, joint_coord, joint_states):
        self.robot = RobotModel(robot_path)
        self.limits = self.robot.phys_limits()
        self.labels = self.limits.index
        if dtype == 'synth':
            self._files = self._parse_synth(joint_coord)
            self._synth_input(joint_coord, self._files)
            self._synth_output(joint_states, self._files)
        else:
            input_ = np.load(joint_coord)
            self.input_ = F.normalize(torch.tensor(input_.reshape((input_.shape[0], -1)), dtype=torch.float32))
            self.output = F.normalize(torch.tensor(self._parse_json(joint_states), dtype=torch.float32))
    
    def _parse_json(self, joint_states):
        json_f = [J for J in os.listdir(joint_states) if J.endswith('.json')]
        dfs = []
        for j in json_f:
            js = pd.read_json(f'{joint_states}/{j}')
            dfs.append(np.array([A['angle'] for A in js[js.columns[0]][0]['joint_angles']], dtype=float))
        return np.array(dfs)[:, :5]

    def _parse_synth(self, path):
        files, coords, angles = [], '', ''
        with open(f'{path}paths.txt') as f:
            while True:
                line = f.readline() 
                if not line:
                    break
                else:
                    files.append(line.rstrip('\n'))
        return files
        
    def _synth_input(self, path, files):
        features = np.empty((0, 15))
        for i in range(len(files)):
            train_set = np.load(f'{path}coords/{files[i]}.npy')
            train_set = train_set.reshape((train_set.shape[0], -1))
            features = np.concatenate((features, train_set), axis=0)
        self.input_ = F.normalize(torch.tensor(np.array(features), dtype=torch.float32))
        
    def _synth_output(self, path, files):
        features = np.empty((0, 6))
        for i in range(len(files)):
            train_set = np.load(f'{path}angles/{files[i]}.npy')
            features = np.concatenate((features, train_set), axis=0)
        self.output = F.normalize(torch.tensor(np.array(features[:, :5]), dtype=torch.float32))    
        
    def __len__(self):
        return len(self.output)

    def __getitem__(self, index):
        return self.input_[index], self.output[index]


class KineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.POSE, self.DIMS, self.OUT = 5, 3, 5
        self.LAYERS = [5, 20]
        # Takes S, L, and U coordinates to output angle L
        self.L = nn.Sequential(
            nn.Linear(self.DIMS * 2, self.DIMS * 2 * self.LAYERS[1]),
            nn.Tanh(),
            nn.Linear(self.DIMS * 2 * self.LAYERS[1], 1),
            nn.Tanh()
        )
        # Takes L, U, and R coordinates and angle L to ouput angle U
        self.U = nn.Sequential(
            nn.Linear(self.DIMS * 3 + 1, (self.DIMS * 3 + 1) * self.LAYERS[1]),
            nn.Tanh(),
            nn.Linear((self.DIMS * 3 + 1) * self.LAYERS[1], 1),
            nn.Tanh()
        )
        # Takes U, R, and B coordinates and angles L, U to output angle R
        self.R = nn.Sequential(
            nn.Linear(self.DIMS * 3 + 2, (self.DIMS * 3 + 2) * self.LAYERS[1]),
            nn.Tanh(),
            nn.Linear((self.DIMS * 3 + 2) * self.LAYERS[1], 1),
            nn.Tanh()
        )
        # Takes R, B coordinates and angles L, U, R to output angle B
        self.B = nn.Sequential(
            nn.Linear(self.DIMS * 3 + 3, (self.DIMS * 3 + 3) * self.LAYERS[1]),
            nn.Tanh(),
            nn.Linear((self.DIMS * 3 + 3) * self.LAYERS[1], 1),
            nn.Tanh()
        )
        # Takes S coordinates and all 5 preceding angles to estimate angle S
        self.S = nn.Sequential(
            nn.Linear(self.DIMS * 2 + 4, (self.DIMS * 2 + 4) * self.LAYERS[1]),
            nn.Tanh(),
            nn.Linear((self.DIMS * 2 + 4) * self.LAYERS[1], 1),
            nn.Tanh()
        )
        self.FC = nn.Sequential(
            nn.Linear(self.OUT, self.OUT * self.LAYERS[1]),
            nn.Tanh(),
            nn.Linear(self.OUT * self.LAYERS[1], self.OUT)
        )

    def forward(self, x):
        theta = self.L(x[:, :6])
        theta = torch.cat((theta, self.U(torch.cat((x[:, :9], theta), 1))), 1)
        theta = torch.cat((theta, self.R(torch.cat((x[:, 3:12], theta), 1))), 1)
        theta = torch.cat((theta, self.B(torch.cat((x[:, 6:15], theta), 1))), 1)
        theta = torch.cat((self.S(torch.cat((x[:, 12:], x[:, :4], theta[:, :3]), 1)), theta), 1)
        return self.FC(theta)

