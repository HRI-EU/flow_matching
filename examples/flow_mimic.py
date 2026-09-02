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
import time

sys.dont_write_bytecode = True
sys.path.append('../external/models')
sys.path.append('../external')
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from unet import ConditionalUnet1D
import collections
from diffusers.training_utils import EMAModel
from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from termcolor import colored
from skvideo.io import vwrite
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.robomimic_replay_lowdim_dataset import RobomimicReplayLowdimDataset
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import pygame

pygame.display.init()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

##################################
dataset_path = os.path.expanduser("../mimic/low_dim_abs.hdf5")

obs_horizon = 1
pred_horizon = 16
action_horizon = 8
action_dim = 20
num_epochs = 4501
vision_feature_dim = 50

# create dataset from file
dataset = RobomimicReplayLowdimDataset(
    dataset_path=dataset_path,
    horizon=pred_horizon,
    abs_action=True,
)

normalizer = dataset.get_normalizer()
normalizers = LinearNormalizer()
normalizers.load_state_dict(normalizer.state_dict())

# create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
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
for epoch in range(0, num_epochs):
    total_loss_train = 0.0
    for data in tqdm(dataloader):
        x_all = normalizers.normalize(data)
        x_img = x_all['obs'][:, :obs_horizon].to(device)
        x_traj = x_all['action'].to(device)
        print(x_img.shape)
        print(x_traj.shape)
        sys.exit(0)

        x_traj = x_traj.float()
        x0 = torch.randn(x_traj.shape, device=device)
        timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

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

    if epoch % 500 == 0:
        ema.copy_to(noise_pred_net.parameters())
        PATH = './flow_ema_%05d.pth' % epoch
        torch.save({'noise_pred_net': noise_pred_net.state_dict(),
                    'epoch': epoch,
                    'ema': ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict()
                    }, PATH)
        ema.restore(noise_pred_net.parameters())

sys.exit(0)

##################################################################
###### test the model
def undo_transform_action(action):
    raw_shape = action.shape
    if raw_shape[-1] == 20:
        # dual arm
        action = action.reshape(-1, 2, 10)

    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3:3 + d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([
        pos, rot, gripper
    ], axis=-1)

    if raw_shape[-1] == 20:
        # dual arm
        uaction = uaction.reshape(*raw_shape[:-1], 14)

    return uaction


PATH = './flow_ema_04500.pth'
state_dict = torch.load(PATH, map_location='cuda')
noise_pred_net.load_state_dict(state_dict['noise_pred_net'])
dim = 500
env_meta = FileUtils.get_env_metadata_from_dataset(
    dataset_path)
env_meta['env_kwargs']['controller_configs']['control_delta'] = False
rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=False,
    render_offscreen=True,
    use_image_obs=False,
)
wrapper = RobomimicLowdimWrapper(
    env=env,
    obs_keys=[
        'object',
        'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
    ],
    render_hw=(dim, dim),
    render_camera_name='frontview'
)

test_start_seed = 1000
n_test = 500
max_steps = 700

scorealllist = []
scorebestlist = []

for epoch in range(n_test):
    seed = test_start_seed + epoch
    scorethislist = []

    for pp in range(10):
        wrapper.seed(seed)

        obs = wrapper.reset()
        obs_deque = collections.deque(
            [obs] * obs_horizon, maxlen=obs_horizon)
        imgs = [wrapper.render(mode='rgb_array')]
        rewards = list()
        done = False
        step_idx = 0

        with tqdm(total=max_steps, desc="Eval Env") as pbar:
            while not done:
                x_img = np.stack([x for x in obs_deque])
                x_img = torch.from_numpy(x_img)
                x_img = normalizers['obs'].normalize(x_img)
                x_img = x_img.to(device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # get image features
                    obs_cond = x_img.flatten(start_dim=1)

                    timehorion = 1
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
                    action_pred = normalizers['action'].unnormalize(naction)

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end, :]

                    for j in range(len(action)):
                        # stepping env
                        env_action = undo_transform_action(action[j])
                        obs, reward, done, info = wrapper.step(env_action)
                        # save observations
                        obs_deque.append(obs)
                        # and reward/vis
                        rewards.append(reward)

                        imgs.append(wrapper.render(mode='rgb_array'))

                        # update progress bar
                        step_idx += 1

                        pbar.update(1)
                        pbar.set_postfix(reward=reward)

                        if step_idx > max_steps or reward == 1:
                            done = True
                        if done:
                            break
