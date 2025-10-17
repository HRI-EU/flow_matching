#
#  Copyright (c) 2024-2025, Honda Research Institute Europe GmbH
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#  notice, this list of conditions and the following disclaimer in the
#  documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  SPDX-License-Identifier: BSD-3-Clause
#
import sys
from termcolor import colored

sys.dont_write_bytecode = True
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
import time


class FlowExpression:
    def __init__(self, model_path="./checkpoint_t/flow_world_p_l=7_x=obj_o=2_h=32_steps_2500.pth", 
                 language_encoder="one_hot"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {self.device}")
        self.PATH = model_path
        state_dict = torch.load(self.PATH, map_location="cuda", weights_only=False)

        action_perams = state_dict["action_perams"]
        self.obs_horizon = action_perams["obs_horizon"]
        self.pred_horizon = action_perams["pred_horizon"]
        print(f"obs_horizon: {self.obs_horizon}, pred_horizon: {self.pred_horizon}")

        self.obs_poses_dim = 29
        self.action_dim = 29

        self.language_encoder = language_encoder
        if self.language_encoder == "t5":
            self.language_feature_dim = 512
            self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
            self.model = T5ForConditionalGeneration.from_pretrained(
                "google-t5/t5-small"
            ).to(self.device)
            print(f"t5 model loaded")
            self.model.eval()
        elif self.language_encoder == "one_hot":
            self.language_feature_dim = 7
        else:
            raise ValueError(
                f"language_encoder {self.language_encoder} not supported, please use 'one_hot' or 't5' "
            )

        self.global_cond_dim = (
            self.language_feature_dim + self.obs_poses_dim
        ) * self.obs_horizon


        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim, global_cond_dim=self.global_cond_dim
        )

        self.nets = nn.ModuleDict({"noise_pred_net": noise_pred_net}).to(self.device)
        self.stats = state_dict["stats"]

        self.nets.noise_pred_net.load_state_dict(state_dict["noise_pred_net"])
        self.nets.noise_pred_net.eval()


    def T5Tokenize(self, input_sample_str: str) -> torch.Tensor:
        t0 = time.time()
        input_ids = self.tokenizer(
            input_sample_str, return_tensors="pt", padding=True
        ).input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model.encoder(input_ids)
        last_hidden_state = outputs.last_hidden_state
        sentence_embedding = last_hidden_state.mean(dim=1)
        t1 = time.time()
        print(f"t5 tokeniz time: {t1 - t0}")
        return sentence_embedding

    def inference(self, input_data: dict, sample_index: int = 0) -> list:
        print(f"device: {self.device}")

        obs_pose = input_data["obs_object_poses"][
            sample_index : sample_index + 1, : self.obs_horizon
        ]

        print(f"one_obs_pose shape: {obs_pose.shape}")
        obs_pose = pusht.normalize_data(obs_pose, stats=self.stats["obs_object_poses"])

        obs_pose_feature = torch.tensor(obs_pose).to(self.device).float()


        if self.language_encoder == "t5":
            obs_lan = input_data["obs_languages"][0][0][0]
            print(f"obs_lan: {obs_lan}")

            sentence_embedding = self.T5Tokenize(obs_lan)
            obs_lan_features = sentence_embedding.unsqueeze(0).expand(
                obs_pose.shape[0], self.obs_horizon, sentence_embedding.shape[-1]
            )
            print(f"one_obs_lan_features shape: {obs_lan_features.shape}")
            print(f"one_obs_pose shape: {obs_pose.shape}")
        else:
            obs_lan_features = pusht.normalize_data(
                input_data["obs_languages"], stats=self.stats["obs_languages"]
            )
            obs_lan_features = torch.tensor(obs_lan_features).to(self.device).float()

        done = False

        while not done:
            with torch.no_grad():
                t1 = time.time()
                
                obs_features = torch.cat([obs_lan_features, obs_pose_feature], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)

                timehorion = 10
                for i in range(timehorion):
                    noise = torch.rand(1, self.pred_horizon, self.action_dim).to(
                        self.device
                    )
                    x0 = noise.expand(obs_cond.shape[0], -1, -1)
                    timestep = torch.tensor([i / timehorion]).to(self.device)

                    if i == 0:
                        vt = self.nets["noise_pred_net"](
                            x0, timestep, global_cond=obs_cond
                        )
                        traj = vt * (1 / timehorion) + x0
                    else:
                        vt = self.nets["noise_pred_net"](
                            traj, timestep, global_cond=obs_cond
                        )
                        traj = vt * (1 / timehorion) + traj

                torch.cuda.synchronize()
                print(colored(f"fm inference time: {time.time() - t1}", "yellow"))

            done = True

            naction = traj.detach().to("cpu").numpy()
            print("naction shape: ", naction.shape)
            naction = naction[0]

            action_pred = pusht.unnormalize_data(naction, self.stats["output_action"])
            # action_pred_dict = {"action": action_pred.tolist()}
            action_pred_list = action_pred.tolist()
            # print(action_pred_list)
            return action_pred_list
