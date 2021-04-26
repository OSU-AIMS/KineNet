## Inverse Kinematics Feed Forward Neural Network
Multi-layer feed-forward neural network for estimating a 6DOF robot's internal joint angles from cartesian coordinates. The neural network is modeled around the concept of how transformation and coordinate matrices feedforward to estimate the angle of each internal joint.

## Installation
Requirements: 
  - `python` version 3.7.6 or later
  - `conda` version 3.18.11 or later
  - `torch` version 1.7.1 or later, `torchvision`, and 'pytorch-pfn-extras'
To install:
``git clone github.com/alexander-jh/KineNet.git``
``conda create --name <env_name> --file requirements.txt``

## Training
Execute command from conda environment.
``python3 src/train.py``

## Generating Training Data
``python3 src/SimDataGenerator.py <# of batches to generate>``.

## Testing
Specify type of data to use:
``python 3 src/test.py``

## Results
Over the training cycle there's an approximate loss of around 10% on average across all 6 joints.
![train](https://github.com/alexander-jh/KineNet/blob/master/src/result/loss.png)

Looking at results by joint, the T-joint is the only joint that's consistently inaccurate.
![joints](https://github.com/alexander-jh/KineNet/blob/master/src/result/train_loss.png)

Mean difference in radians between joint estimation and actual.
| S-Joint | L-Joint | U-Joint | R-Joint | B-Joint |
| :-----: | :-----: | :-----: | :-----: | :-----: |
| -0.0035 |  0.0075 | 0.0042  | -0.0129 |  0.0036 |

