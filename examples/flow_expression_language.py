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
# SPDX-License-Identifier: BSD-3-Clause
#
# This notebook is an example of "Affordance-based Robot Manipulation with Flow Matching" https://arxiv.org/abs/2409.01083

import sys
from termcolor import colored

sys.dont_write_bytecode = True
sys.path.append(r"C:\Users\HoloAdmin\Desktop\OTHERS\flow_matching\external\models")
import numpy as np
import torch

# import pusht
import torch.nn as nn
from tqdm import tqdm
from unet import ConditionalUnet1D
from resnet import get_resnet
from resnet import replace_bn_with_gn
import collections
from diffusers.training_utils import EMAModel
from torch.utils.data import Dataset, DataLoader
from diffusers.optimization import get_scheduler
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

##################################
########## download the pusht data and put in the folder
dataset_path = "pusht_cchi_v7_replay.zarr.zip"

obs_horizon = 1
pred_horizon = 1
action_dim = 2053
action_horizon = 1
num_epochs = 3001
vision_feature_dim = 1536


class ExpressionDataset(Dataset):
    def __init__(self):
        self.train_data_x = np.load("fm_train_data_x.npy")
        self.train_data_y = np.load("fm_train_data_y.npy")

    def __len__(self):
        return len(self.train_data_x)

    def __getitem__(self, idx):
        # TODO: normalize data

        return {"x": self.train_data_x[idx], "y": self.train_data_y[idx]}


expression_dataset = ExpressionDataset()

# create dataset from file
# dataset = pusht.PushTImageDataset(
#     dataset_path=dataset_path,
#     pred_horizon=pred_horizon,
#     obs_horizon=obs_horizon,
#     action_horizon=action_horizon,
# )
# save training data statistics (min, max) for each dim
# stats = dataset.stats

# create dataloader
dataloader = DataLoader(
    expression_dataset,
    batch_size=5,
    num_workers=0,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=False,
)

##################################################################
# create network object
vision_encoder = get_resnet("resnet18")
vision_encoder = replace_bn_with_gn(vision_encoder)
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim, global_cond_dim=vision_feature_dim
)

# noise_pred_net = ConditionalUnet1D(
#     input_dim=expression_dim, global_cond_dim=robot_behaviour_dim
# )

nets = nn.ModuleDict(
    {"vision_encoder": vision_encoder, "noise_pred_net": noise_pred_net}
).to(device)

##################################################################
sigma = 0.0
ema = EMAModel(parameters=nets.parameters(), power=0.75)
optimizer = torch.optim.AdamW(params=nets.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name="cosine",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs,
)

FM = ConditionalFlowMatcher(sigma=sigma)

avg_loss_train_list = []


########################################################################
#### Train the model
def train():
    for epoch in range(num_epochs):
        total_loss_train = 0.0
        for data in tqdm(dataloader):
            # x_img = data["image"][:, :obs_horizon].to(device)
            # x_pos = data["agent_pos"][:, :obs_horizon].to(device)
            # x_traj = data["action"].to(device)

            x_robot_action = data["x"].to(device)
            x_expression = data["y"].to(device).unsqueeze(dim=1)
            x_expression = x_expression.float()

            # x_traj = x_traj.float()
            # x0 = torch.randn(x_traj.shape, device=device)
            x0 = torch.randn(x_expression.shape, device=device)
            print("x_expression shape:")
            print(x_expression.shape)

            # timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)
            timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_expression)

            print("xt, ut, timestep shape:")
            print(xt.shape, ut.shape, timestep.shape)

            # encoder vision features
            # image_features = nets["vision_encoder"](x_img.flatten(end_dim=1))
            # image_features = image_features.reshape(*x_img.shape[:2], -1)
            # obs_features = torch.cat([image_features, x_pos], dim=-1)
            # obs_cond = obs_features.flatten(start_dim=1)
            obs_cond = x_robot_action
            print("obs_cond shape: ")
            print(obs_cond.shape)

            vt = nets["noise_pred_net"](xt, timestep, global_cond=obs_cond)

            loss = torch.mean((vt - ut) ** 2)
            total_loss_train += loss.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            # update Exponential Moving Average of the model weights
            ema.step(nets.parameters())

        avg_loss_train = total_loss_train / len(dataloader)
        avg_loss_train_list.append(avg_loss_train.detach().cpu().numpy())
        print(
            colored(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}", "yellow")
        )

        if epoch == 3000:
            ema.copy_to(nets.parameters())
            PATH = "./checkpoint_t/flow_ema_%05d.pth" % epoch
            torch.save(
                {
                    "vision_encoder": nets.vision_encoder.state_dict(),
                    "noise_pred_net": nets.noise_pred_net.state_dict(),
                },
                PATH,
            )


########################################################################
###### test the model
def test():
    PATH = "/home/hri/z_flow/checkpoint_t/flow_ema_03000.pth"
    state_dict = torch.load(PATH, map_location="cuda")
    ema_nets = nets
    ema_nets.vision_encoder.load_state_dict(state_dict["vision_encoder"])
    ema_nets.noise_pred_net.load_state_dict(state_dict["noise_pred_net"])

    max_steps = 300
    env = pusht.PushTImageEnv()

    test_start_seed = 1000
    n_test = 500

    ###### please choose the seed you want to test
    for epoch in range(n_test):
        seed = test_start_seed + epoch
        env.seed(seed)

        for pp in range(10):
            obs, info = env.reset()
            obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
            imgs = [env.render(mode="rgb_array")]
            rewards = list()
            done = False
            step_idx = 0

            with tqdm(total=max_steps, desc="Eval PushTImageEnv") as pbar:
                while not done:
                    B = 1
                    x_img = np.stack([x["image"] for x in obs_deque])
                    x_pos = np.stack([x["agent_pos"] for x in obs_deque])
                    x_pos = pusht.normalize_data(x_pos, stats=stats["agent_pos"])

                    x_img = torch.from_numpy(x_img).to(device, dtype=torch.float32)
                    x_pos = torch.from_numpy(x_pos).to(device, dtype=torch.float32)
                    # infer action
                    with torch.no_grad():
                        # get image features
                        # t1 = time.time()
                        image_features = ema_nets["vision_encoder"](x_img)
                        obs_features = torch.cat([image_features, x_pos], dim=-1)
                        obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                        timehorion = 1
                        for i in range(timehorion):
                            noise = torch.rand(1, pred_horizon, action_dim).to(device)
                            x0 = noise.expand(x_img.shape[0], -1, -1)
                            timestep = torch.tensor([i / timehorion]).to(device)

                            if i == 0:
                                vt = nets["noise_pred_net"](
                                    x0, timestep, global_cond=obs_cond
                                )
                                traj = vt * 1 / timehorion + x0

                            else:
                                vt = nets["noise_pred_net"](
                                    traj, timestep, global_cond=obs_cond
                                )
                                traj = vt * 1 / timehorion + traj

                    # print(time.time() - t1)

                    naction = traj.detach().to("cpu").numpy()
                    naction = naction[0]
                    action_pred = pusht.unnormalize_data(naction, stats=stats["action"])

                    # only take action_horizon number of actions
                    start = obs_horizon - 1
                    end = start + action_horizon
                    action = action_pred[start:end, :]

                    # x_img = x_img[0, :].permute((1, 2, 0))
                    # plot_trajectory(x0[0].detach().cpu().numpy(), vt[0].detach().cpu().numpy(),
                    #                 action_pred,
                    #                 x_img.detach().cpu().numpy())

                    # execute action_horizon number of steps
                    for j in range(len(action)):
                        # stepping env
                        obs, reward, done, _, info = env.step(action[j])
                        # save observations
                        obs_deque.append(obs)
                        # and reward/vis
                        rewards.append(reward)
                        imgs.append(env.render(mode="rgb_array"))

                        # update progress bar
                        step_idx += 1

                        pbar.update(1)
                        pbar.set_postfix(reward=reward)

                        if step_idx > max_steps:
                            done = True
                        if done:
                            break


if __name__ == "__main__":
    # Check if an argument was provided
    if len(sys.argv) < 2:
        print("No argument provided. Please specify 'train', 'test', or 'print'.")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == "train":
        train()
    elif arg == "test":
        test()
    elif arg == "unittest":
        print("Uni Test Successful")
    else:
        print(f"Unknown argument '{arg}'. Please specify 'train', 'test', or 'print'.")
