import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from RobotModel import RobotModel


class KNData(Dataset):
    """
    Dataloader class for train/test data import. Extends torch Dataset class.
    :var self.input_: Loaded input torch tensor for joint coordinates
    :type self.input_: torch tensor - dims = [(batch) x (joint coordinates) x 3]
    :var self.output: Loaded output torch tensor for validation position
    :type self.output: torch tensor - dims = [(batch) x (joint positions)]
    """

    def __init__(self, dtype, robot_path, joint_coord, joint_states):
        """
        Default constructor. Constructs batched, loadable train and test tensors.
        :param dtype: Specifies if real or synthetic data is to be used
        :type dtype: string
        :param robot_path: file path to robot URDF
        :type robot_path: string
        :param joint_coord: file path to joint coordinates
        :type joint_coord: string
        :param joint_states: file path to joint angles
        :type joint_states: string
        """
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

    @staticmethod
    def _parse_json(joint_states):
        """
        Reads JSON to extract joint angles from real data training set. Merges all json's at location into single
        array.
        :param joint_states: location to json directory
        :type joint_states: string
        :return: compiled array of joint angles
        :rtype: numpy array - dims = [(batch) x (angles - T)]
        """
        json_f = [J for J in os.listdir(joint_states) if J.endswith('.json')]
        dfs = []
        for j in json_f:
            js = pd.read_json(f'{joint_states}/{j}')
            dfs.append(np.array([A['angle'] for A in js[js.columns[0]][0]['joint_angles']], dtype=float))
        return np.array(dfs)[:, :5]

    @staticmethod
    def _parse_synth(path):
        """
        Constructs list of all data files of synthetic data specified in path training directory file.
        :param path: Location of path.txt
        :type path: string
        :return: all file locations serialized in path.txt
        :rtype: list of strings
        """
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
        """
        Handles reading of synthetic input data. Normalizes dataframe to [-1,1]
        :param path: file path location to synthetic data
        :type path: string
        :param files: all serialized files in path
        :type files: list of string
        """
        features = np.empty((0, 15))
        for i in range(len(files)):
            train_set = np.load(f'{path}coords/{files[i]}.npy')
            train_set = train_set.reshape((train_set.shape[0], -1))
            features = np.concatenate((features, train_set), axis=0)
        self.input_ = F.normalize(torch.tensor(np.array(features), dtype=torch.float32))

    def _synth_output(self, path, files):
        """
        Handles reading of synthetic output data. Normalizes angles to fit in [-1,1]
        :param path: file location of output validation angles
        :type path: string
        :param files: serialized list of all files in path
        :type files: list of string
        """
        features = np.empty((0, 6))
        for i in range(len(files)):
            train_set = np.load(f'{path}angles/{files[i]}.npy')
            features = np.concatenate((features, train_set), axis=0)
        self.output = F.normalize(torch.tensor(np.array(features[:, :5]), dtype=torch.float32))

    def __len__(self):
        """
        Mandatory override function in superclass. Returns length of loaded dataset.
        :return: entry size of dataset
        :rtype: long
        """
        return len(self.output)

    def __getitem__(self, index):
        """
        Mandatory override function in superclass. Retrieves dataset entry from tensors.
        :param index: batch location identifier
        :type index: long
        :return: input, output tensor for given entry
        :rtype: tuple of two torch.tensors - dims = ([1 x (joints) x 3], [1 x (joints - T)])
        """
        return self.input_[index], self.output[index]


class KineNet(nn.Module):
    """
    Neural net module for predicting output joint angles from internal inputs. Extends torch neural
    net module class.
    """
    def __init__(self):
        """
        Neural net model.
        """
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
        """
        Mandatory override function in nn.Module. Applies forward actions to data tensor.
        :param x: input joint coordinate training data
        :type x: torch tensor - dims = [(batch size) x (joints) x 3]
        :return: joint angle estimates
        :rtype: torch tensor - dims = [(batch size) x (joints - T)]
        """
        theta = self.L(x[:, :6])
        theta = torch.cat((theta, self.U(torch.cat((x[:, :9], theta), 1))), 1)
        theta = torch.cat((theta, self.R(torch.cat((x[:, 3:12], theta), 1))), 1)
        theta = torch.cat((theta, self.B(torch.cat((x[:, 6:15], theta), 1))), 1)
        # Prepends T to the angle vector theta to maintain order S,L,U,R,B
        theta = torch.cat((self.S(torch.cat((x[:, 12:], x[:, :4], theta[:, :3]), 1)), theta), 1)
        return self.FC(theta)
