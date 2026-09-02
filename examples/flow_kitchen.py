#!/usr/bin/env python
#
# Copyright (c) 2024, Honda Research Institute Europe GmbH
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This notebook is an example of "Affordance-based Robot Manipulation with Flow Matching" https://arxiv.org/abs/2409.01083

import sys

sys.dont_write_bytecode = True
sys.path.append('../external/models')
sys.path.append('../external')
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import torch.nn as nn
from torchvision import datasets, transforms
from tqdm import tqdm
from unet import ConditionalUnet1D
from resnet import get_resnet
from resnet import replace_bn_with_gn
import collections
from diffusers.training_utils import EMAModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from termcolor import colored
import torchdiffeq
import torchsde
from torchdyn.core import NeuralODE
import pathlib
from skvideo.io import vwrite
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
import kitchen_lowdim_dataset
from diffusion_policy.env.kitchen.v0 import KitchenAllV0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

##################################
###### add Franka kitchen data to some folder
dataset_path = "./kitchen/data"

obs_horizon = 1
pred_horizon = 16
action_dim = 9
action_horizon = 8
num_epochs = 4501
vision_feature_dim = 60

# create dataset from file
dataset = kitchen_lowdim_dataset.KitchenLowdimDataset(
    dataset_dir=dataset_path,
    horizon=16,
)
print(len(dataset))

# create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

##################################################################
# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=vision_feature_dim
).to(device)

##################################################################
sigma = 0.0
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)
optimizer = torch.optim.AdamW(params=noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

FM = ConditionalFlowMatcher(sigma=sigma)
avg_loss_train_list = []
avg_loss_val_list = []

########################################################################
#### Train the model
for epoch in range(num_epochs):
    total_loss_train = 0.0
    for data in tqdm(dataloader):
        x_img = data['obs'][:, :obs_horizon].to(device)
        x_traj = data['action'].to(device)

        x_traj = x_traj.float()
        x0 = torch.randn(x_traj.shape, device=device)
        timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

        # encoder state features
        obs_cond = x_img.flatten(start_dim=1)
        vt = noise_pred_net(xt, timestep, global_cond=obs_cond)

        loss = torch.mean((vt - ut) ** 2)
        total_loss_train += loss.detach()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        # update Exponential Moving Average of the model weights
        ema.step(noise_pred_net.parameters())

    avg_loss_train = total_loss_train / len(dataloader)
    avg_loss_train_list.append(avg_loss_train.detach().cpu().numpy())
    print(colored(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}", 'yellow'))

    if epoch == 4500:
        ema.copy_to(noise_pred_net.parameters())
        PATH = './checkpoint_k/flow_ema_%05d.pth' % epoch
        torch.save({
            'noise_pred_net': noise_pred_net.state_dict(),
        }, PATH)
         ema.restore(noise_pred_net.parameters())

sys.exit(0)

##################################################################
###### test the model
PATH = './flow_ema_04500.pth'
state_dict = torch.load(PATH, map_location='cuda')
noise_pred_net.load_state_dict(state_dict['noise_pred_net'])

max_steps = 280
env = KitchenAllV0(use_abs_action=False)

test_start_seed = 10000
n_test = 500

###### please choose the seed you want to test
for epoch in range(n_test):
    seed = test_start_seed + epoch
    env.seed(seed)

    for pp in range(10):
        obs = env.reset()

        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        imgs = [env.render(mode='rgb_array')]
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc="Eval KitchenAllV0") as pbar:
            while not done:
                x_img = np.stack([x for x in obs_deque])
                x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # get image features
                    obs_cond = x_img.flatten(start_dim=1)

                    timehorion = 16
                    for i in range(timehorion):
                        noise = torch.rand(1, pred_horizon, action_dim).to(device)
                        x0 = noise.expand(x_img.shape[0], -1, -1)
                        timestep = torch.tensor([i / timehorion]).to(device)

                        if i == 0:
                            vt = noise_pred_net(x0, timestep, global_cond=obs_cond)
                            traj = (vt * 1 / timehorion + x0)

                        else:
                            vt = noise_pred_net(traj, timestep, global_cond=obs_cond)
                            traj = (vt * 1 / timehorion + traj)

                    naction = traj.detach().to('cpu').numpy()
                    naction = naction[0]
                    action_pred = naction

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end, :]

                    for j in range(len(action)):
                        # stepping env
                        obs, reward, done, info = env.step(action[j])
                        # save observations
                        obs_deque.append(obs)
                        # and reward/vis
                        rewards.append(reward)
                        imgs.append(env.render(mode='rgb_array'))

                        # update progress bar
                        step_idx += 1

                        pbar.update(1)
                        pbar.set_postfix(reward=reward)

                        if step_idx > max_steps or sum(rewards) == 4:
                            done = True
                        if done:
                            break
