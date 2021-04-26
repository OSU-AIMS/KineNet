## Inverse Kinematics Feed Forward Neural Network
Multi-layer feed-forward neural network for estimating a 6DOF robot's internal joint angles from cartesian coordinates. The neural network is modeled around the concept of how transformation and coordinate matrices feedforward to estimate the angle of each internal joint.
![nn](https://github.com/alexander-jh/KineNet/blob/main/src/result/nn_diagram.png)

## Installation
Requirements: `torch` version 1.7.1 or later, `torchvision`, and 'pytorch-pfn-extras'.

## Training
Specify whether to use real or synthetic data.
``python3 src/train.py <real/synth>``

## Generating Training Data
``python3 src/SimDataGenerator.py <# of batches to generate>``.

## Testing
Specify type of data to use:
``python 3 src/test.py <real/synth>``.

## Results
Over the training cycle there's an approximate loss of around 10% on average across all 6 joints.
![train](https://github.com/alexander-jh/KineNet/blob/main/src/result/loss.png)

Looking at results by joint, the T-joint is the only joint that's consistently inaccurate.
![joints](https://github.com/alexander-jh/KineNet/blob/main/src/result/train_loss.png)

Mean difference in radians between joint estimation and actual.
| S-Joint | L-Joint | U-Joint | R-Joint | B-Joint | T-Joint |
| :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
| -0.0035 |  0.0075 | 0.0042  | -0.0129 |  0.0036 |  0.0161 |

