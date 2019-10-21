# DDPG with Hindsight Experience Replay (HER)

This is the PyTorch implementation of DDPG with Hindsight Experience Replay (HER).
The code is adapted from [@julianalverio's reimplementation](https://github.com/julianalverio/ddpg_her/). For more details, please check the [original paper](https://arxiv.org/abs/1707.01495).

## Getting Started

You can take the following steps to run DDPG (with HER) on the OpenAI gym Fetch environments.

### Setting up a conda environment

```bash
conda create tensorflow mpi4py python=3.6 -n ddpg
conda activate ddpg
```

### Installing the required packages

```bash
LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH pip install mujoco_py==2.0.2.4 torch tensorboard==1.14.0 opencv-python scipy GPUtil cloudpickle requests future gym[all]
```

### Training the agent

After setting up the packages, you can run `run.py` as follows to train the agent.

```bash
LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin:$LD_LIBRARY_PATH python run.py --num_timesteps 10000000 --num_workers 25 --env FetchPickAndPlace-v1 --replay_strategy future
```

There are some parameters you can set.
* `env`: The OpenAI gym Fetch environment to use. 
* `replay_strategy`: The replay strategy to use. You need to set it to `future` to use HER, and `none` to use the original DDPG strategy.
* `num_workers`: The number of workers to run the rollout.

### Results

After training, you will get learning curves similar to the figure below. With HER, the agent in FetchPickAndPlace-v1 can reach score 0.9 in around 400 epochs. Without HER, i.e. the original DDPG, the agent's score stays at around 0.03 and 0.04, which means the agent doesn't learn anything.

<p align="center">
  <img src="https://github.com/ylkuo/ddpg/blob/master/figures/fetch-pickandplace.png?raw=true"/><br />
  Training results for Fetch Pick and Place task. <br/>(red) DDPG with HER, (blue) original DDPG
</p>
