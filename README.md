# MQN Offline

## Introduction

This is a Pytorch implementation for our paper on 
"Monotonic Quantile Network for Worst-Case Offline Reinforcement Learning"

## Prerequisites

- Python3.6 or 3.7 with pytorch 1.8
- [D4RL](https://github.com/rail-berkeley/d4rl) with v2 [dataset](http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2_old/) 
- OpenAI [Gym](http://gym.openai.com/) with [mujoco-py](https://github.com/openai/mujoco-py)

## Installation and Usage

For running MQN-offline on the D4RL environments, run:

```
python train_offline.py --env walker2d-medium-v0 --gpu 0
```

For running MQN-offline on the risk-sensitive D4RL environments, run:

```
python train_offline_cvar.py --env walker2d-medium-v0 --gpu 0
```

The core implementation is given in `distributional/wcql.py`

## Execution

The data for separate runs is stored on disk under the result directory 
with filename `<env-id>/<seed>-<timestamp>/`. Each run directory contains

- `eval.csv` Record the rewards in evaluation.
- `progress.csv` Record the losses, rewards, Q-values in training.
- `args.json` The hyper-parameters in training.
- `models` The saved model weights.

In case of any questions, bugs, suggestions or improvements, please feel free to open an issue.
