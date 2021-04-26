import numpy as np
import pandas as pd
import os
import sys
from ForwardKinematicSolverMH5L import FwdKinematic_MH5L_AllJoints as fwdk
from RobotModel import RobotModel

def get_urdf(model = None):
    if type(model) is None:
        model = input('Enter URDF file name in data/urdf/: ')
    return RobotModel(model)

def rand_pos(min, max):
    return lambda : np.random.uniform(min, max, 1)

def retrieve_joint_limits(model):
    limits = model.phys_limits()
    limits = limits[limits.columns[1:]].iloc[1:].to_numpy()
    return [rand_pos(limits[l, 0], limits[l, 1]) for l in range(len(limits))]

def rand_angle_vector(limits):
    return [float(R()) for R in limits]

def get_last_entry(data_dir):
    max_entry = 0
    with open(f'{data_dir}/paths.txt') as f:
        while True:
            line = f.readline() 
            if not line:
                break
            elif int(line) > max_entry:
                max_entry = int(line)
    return max_entry

def save_results(angles, coords, data_dir):
    last = zpad(get_last_entry(data_dir) + 1)
    np.save(f'{data_dir}/angles/{last}.npy', angles)
    np.save(f'{data_dir}/coords/{last}.npy', coords)
    f = open(f'{data_dir}/paths.txt', 'a')
    f.write(f'{last}\n')
    f.close()      
    
def zpad(num):
    return "{:07d}".format(num)

def main(batch=500):   
  data_dir = '../data/train/synth'
  model = get_urdf('../data/urdf/mh5l.urdf')
  size = int(sys.argv[1])
  limits = retrieve_joint_limits(model)
  for i in range(size):
    angles = np.array([rand_angle_vector(limits) for i in range(batch)])
    coords = np.array([fwdk(angles[i, :])[1:] for i in range(batch)])
    save_results(angles, coords, data_dir)


if __name__ == "__main__":
    main()
