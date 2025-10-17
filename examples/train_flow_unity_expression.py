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

import os
import sys
from termcolor import colored

sys.dont_write_bytecode = True
# sys.path.append('../models')
sys.path.append(r"C:\Users\Public\ChaosProjects\flow_matching\external\models")
sys.path.append(r"C:\Users\Public\ChaosProjects\flow_matching\external")

import numpy as np
import torch
import pusht
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

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from torch.utils.tensorboard import SummaryWriter
import time
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

##################################
########## download the pusht data and put in the folder
dataset_path = r"C:\Users\Public\ChaosProjects\flow_matching\unity_data_20250510_141653_n=70363_obpose=29_act=29_world.npz" 

obs_horizon = 2
pred_horizon = 32
action_horizon = 8
num_epochs = 5001
# vision_feature_dim = 514

obs_poses_dim = 29
language_feature_dim = 7
action_dim = 29

global_cond_dim = (language_feature_dim + obs_poses_dim) * obs_horizon
avg_loss_train_list = []

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small").to(device)


# dataset
class UnityExpressionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
    ):
        # read from zarr dataset
        # dataset_root = zarr.open(dataset_path, 'r')
        dataset_root = np.load(dataset_path)
        episode_ends = dataset_root["end_idx"][:]

        train_data = {
            "obs_object_poses": dataset_root["OBS_POSES"][:],
            "obs_languages": dataset_root["OBS_LANG"][:],
            "output_action": dataset_root["ACTIONS"][:],
        }

        # compute start and end of each state-action sequence
        # also handles padding
        indices = pusht.create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            # if key == "obs_languages":
            #     # language data is already normalized
            #     stats[key] = {"min": -1.0, "max": 1.0}
            #     normalized_train_data[key] = data
            #     continue
            stats[key] = pusht.get_data_stats(data)
            normalized_train_data[key] = pusht.normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )

        # get nomralized data using these indices
        nsample = pusht.sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["obs_object_poses"] = nsample["obs_object_poses"][: self.obs_horizon, :]
        nsample["obs_languages"] = nsample["obs_languages"][: self.obs_horizon, :]
        # print(f"nsample['obs_object_poses']: {nsample['obs_object_poses']}")
        # print(f"nsample['obs_languages']: {nsample['obs_languages']}")

        return nsample


# create dataset from file
dataset = UnityExpressionDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon,
)
# save training data statistics (min, max) for each dim
stats = dataset.stats


# create dataloader
####### custom collate function to handle different data types
def custom_collate_fn(batch):
    # batch is a list, each element is a dict returned by __getitem__
    collated = {}
    # 数值型字段直接stack
    collated["obs_object_poses"] = torch.stack(
        [torch.from_numpy(b["obs_object_poses"]) for b in batch], dim=0
    )
    collated["obs_languages"] = torch.stack(
        [torch.from_numpy(b["obs_languages"]) for b in batch], dim=0
    )
    if "output_action" in batch[0]:
        collated["output_action"] = torch.stack(
            [torch.from_numpy(b["output_action"]) for b in batch], dim=0
        )
    return collated


dataloader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process after each epoch
    persistent_workers=True,
    collate_fn=custom_collate_fn,
)

##################################################################
# load the t5 model


def T5Tokenize(input_sample_str: str) -> torch.Tensor:
    # print(f"""input_sample_str: {input_sample_str}""")

    # Tokenize a batch of strings with padding and truncation
    input_ids = tokenizer(input_sample_str, return_tensors="pt", padding=True).input_ids

    # Move data to GPU
    input_ids = input_ids.to(device)

    t1 = time.time()

    # Get encoder outputs without gradient computation
    with torch.no_grad():
        outputs = model.encoder(input_ids)

    # Last hidden state from the encoder
    last_hidden_state = outputs.last_hidden_state

    # Compute sentence embeddings by averaging over the sequence length dimension
    sentence_embedding = last_hidden_state.mean(dim=1)

    # Print debug information
    # print("Token Embeddings Shape:", last_hidden_state.shape)
    # print("Sentence Embedding Shape:", sentence_embedding.shape)

    t2 = time.time()
    # print("encoding time: ", (t2 - t1))
    return sentence_embedding  # (batch, 512)


##################################################################
# create network object
vision_encoder = get_resnet("resnet18")
vision_encoder = replace_bn_with_gn(vision_encoder)
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim, global_cond_dim=global_cond_dim
)
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


########################################################################
#### Train the model
def train():
    start_epoch = 0
    resume_path = ""  # change as needed

    resume_path = "./checkpoint_t/flow_world_p_l=7_x=obj_o=2_h=32_steps_1500.pth"  # if you want to resume training
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        nets["vision_encoder"].load_state_dict(checkpoint["vision_encoder"])
        nets["noise_pred_net"].load_state_dict(checkpoint["noise_pred_net"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # resume from the next epoch
        print(f"Loaded checkpoint from: {resume_path}")

    writer = SummaryWriter(log_dir=f"runs/flow_unity_expression_o={obs_horizon}_h={pred_horizon}")

    for epoch in range(start_epoch, num_epochs):
        total_loss_train = 0.0

        for data in tqdm(dataloader):

            obs_pose = data["obs_object_poses"][:, :obs_horizon].to(device)
            obs_pose = obs_pose.float()
            x_traj = data["output_action"].to(device)
            x_traj = x_traj.float()

            # print(f"obs_pose shape: {obs_pose.shape}")
            # print(f"x_traj shape: {x_traj.shape}")

            # Convert list of language lists to a list of strings.
            # as each epsoide only has one language, we don't need to T5Tokenize for each data point.
            obs_lan_features = data["obs_languages"].to(device)
            obs_lan_features = obs_lan_features.float()
            # print(f"\n obs_lan: {obs_lan}")
            # encoder language features
            # sentence_embedding = T5Tokenize(obs_lan)

            # after tokenization, sentence_embedding is already on device
            # obs_lan_features = sentence_embedding.unsqueeze(0).expand(
            #     obs_pose.shape[0], obs_horizon, sentence_embedding.shape[-1]
            # )

            # print(f"obs_lan_features: {obs_lan_features}")
            # print(f"obs_lan_features shape: {obs_lan_features.shape}")

            x0 = torch.randn(x_traj.shape, device=device)
            timestep, xt, ut = FM.sample_location_and_conditional_flow(x0, x_traj)

            # merge language and position features
            obs_features = torch.cat([obs_lan_features, obs_pose], dim=-1)
            obs_cond = obs_features.flatten(start_dim=1)

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
        writer.add_scalar("Loss/train", avg_loss_train, epoch)
        print(
            colored(f"epoch: {epoch:>02},  loss_train: {avg_loss_train:.10f}", "yellow")
        )

        if epoch % 500 == 0:
            ema.copy_to(nets.parameters())
            # PATH = "./checkpoint_t/flow_world_p_l=7_x=obj_o=2_h=16_steps_%05d.pth" % epoch
            PATH = f"""./checkpoint_t/flow_world_p_l=7_x=obj_o={obs_horizon}_h={pred_horizon}_steps_{epoch}.pth"""
            torch.save(
                {
                    "epoch": epoch,
                    "vision_encoder": nets.vision_encoder.state_dict(),
                    "noise_pred_net": nets.noise_pred_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "stats": stats,
                    "action_perams": {
                        "obs_horizon": obs_horizon,
                        "pred_horizon": pred_horizon,
                        "action_horizon": action_horizon,
                    },
                },
                PATH,
            )
    writer.close()


def print_one_batch_data():
    data = next(iter(dataloader))
    # print(f"""data["obs_object_poses"].shape: {data["obs_object_poses"].shape}""")
    print(f"""data["obs_object_poses"]: {data["obs_object_poses"]}""")
    print(f"""data["obs_languages"]: {data["obs_languages"]}""")

    # obs_pose = data["obs_object_poses"][:, :obs_horizon]
    # obs_pose = obs_pose.float()
    # one_obs_pose = obs_pose[0]
    # print(f"one_obs_pose shape: {one_obs_pose.shape}")
    # obs_pose = pusht.normalize_data(obs_pose, stats=stats["obs_object_poses"])

    # obs_lan = data["obs_languages"][0]

    # print(f"""data["obs_languages"][0]: {data["obs_languages"][0]}""")


if __name__ == "__main__":
    # Check if an argument was provided
    if len(sys.argv) < 2:
        print("No argument provided. Please specify 'train', 'test', or 'print_data'.")
        sys.exit(1)

    arg = sys.argv[1].lower()

    if arg == "train":
        train()
    elif arg == "print_data":
        print_one_batch_data()

    elif arg == "unittest":
        print("Uni Test Successful")
    else:
        print(f"Unknown argument '{arg}'. Please specify 'train', 'test', or 'print'.")
