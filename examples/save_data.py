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
import json
import os
import sys
import numpy as np
from datetime import datetime


class SaveData:
    class OneHotActionEncoder:
        ACTION_NAMES = ["calm", "angry", "sad", "happy",  "curious", "fear", "bored",] # model one hot encoding

        @classmethod
        def encode(cls, action: str):
            vec = [0.0] * len(cls.ACTION_NAMES)
            if action in cls.ACTION_NAMES:
                vec[cls.ACTION_NAMES.index(action)] = 1.0
            return vec

        @classmethod
        def decode(cls, vec):
            idx = int(np.argmax(vec))
            if 0 <= idx < len(cls.ACTION_NAMES):
                return cls.ACTION_NAMES[idx]
            return None

    def __init__(self):
        self.data = []
        self.filename = "unity_data"
        self.facial_expression_names = [
            "EyesClosedL",
            "EyesClosedR",
            "DimplerL",
            "DimplerR",
            "ChinRaiserB",
            "BrowLowererL",
            "BrowLowererR",
        ]

    def update(self, data: dict):
        self.data.append(data)

    def renew_dats(self):
        self.data = []

    def save_as_json(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("data/unity_data", f"{self.filename}_{timestamp}.jsonl")
        with open(filename, "w") as f:
            for item in self.data:
                f.write(json.dumps(item) + "\n")
        print("save data to", filename)

    def save_json_into_npz(self):
        data_folder = "data/unity_world_pose_data"
        episodes_obs_poses = []
        episodes_obs_lang = []
        episodes_actions = []
        episodes_end_idx = []

        _epi_start_idx = -1

        for filename in os.listdir(data_folder):
            if filename.endswith(".jsonl"):
                with open(os.path.join(data_folder, filename), "r") as f:
                    dicts_list = []
                    for line in f:
                        dicts_list.append(json.loads(line.strip()))

                    (obs_poses, obs_lang, actions, epi_len) = self.flatten_data(
                        dicts_list
                    )
                    # (N, 42), (N, 28), int
                    episodes_obs_poses += obs_poses
                    episodes_obs_lang += obs_lang
                    episodes_actions += actions

                    _epi_start_idx += epi_len
                    episodes_end_idx.append(_epi_start_idx)

        OBS_POSES = np.array(episodes_obs_poses)
        OBS_LANG = np.array(episodes_obs_lang)
        ACTIONS = np.array(episodes_actions)

        end_idx = np.array(episodes_end_idx)
        print(
            f"""OBS_POSES shape: {OBS_POSES.shape}, OBS_LANG shape: {OBS_LANG.shape}, ACTION shape: {ACTIONS.shape}, end_idx shape: {end_idx.shape}"""
        )

        # Precompute decoded actions and count occurrences
        action_counts = {name: 0 for name in self.OneHotActionEncoder.ACTION_NAMES}
        for obs in episodes_obs_lang:
            decoded_action = self.OneHotActionEncoder.decode(obs)
            if decoded_action in action_counts:
                action_counts[decoded_action] += 1

        # Print the data points for each action name
        for name, count in action_counts.items():
            print(f"{name} data points: {count}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        np_filename = f"""{self.filename}_{timestamp}_n={OBS_POSES.shape[0]}_obpose={OBS_POSES.shape[1]}_act={ACTIONS.shape[1]}_world.npz"""

        np.savez(
            np_filename,
            OBS_POSES=OBS_POSES,
            OBS_LANG=OBS_LANG,
            ACTIONS=ACTIONS,
            end_idx=end_idx,
        )
        print(f"Data saved to {np_filename}")

    # @staticmethod
    def flatten_data(self, d_list: list) -> tuple:
        X_POSES = []
        X_LANG = []
        Y_ACTION = []

        for i, d in enumerate(d_list):
            _x_lang = []
            _x = []
            _y = []

            ######### Input X #########
            # add x in order of TargetObject, ActionName
            for k in ["TargetObject"]:
                _x += self._pose_dict_to_list(d[k], withPosition=True)

            # add agent pose in order of (LeftHand, RightHand)
            for k in ["LeftHand", "RightHand"]:
                _x += self._pose_dict_to_list(d[k], withPosition=True)

            # add agent pose in order of (Head, Eye)
            for k in [
                "Head",
                "Eye",
            ]:
                _x += self._pose_dict_to_list(d[k], withPosition=False)

            # the action name (now as one-hot)
            for k in ["ActionName"]:
                _x_lang += self.one_hot_action(d[k])

            ######### Output Y #########
            # add y in order of (Head, LeftHand, RightHand, FacialExpression
            for k in [
                "Head",
                "Eye",
            ]:
                _y += self._pose_dict_to_list(d[k], withPosition=False)

            for k in ["LeftHand", "RightHand"]:
                _y += self._pose_dict_to_list(d[k], withPosition=True)

            # according to the order of the self.facial_expression_names to flatten
            for face_k in self.facial_expression_names:
                _y.append(d["FacialExpression"][face_k])

            ##################
            # print(f"len of _x: {len(_x)}")
            # print(f"len of _x_lang:{len(_x_lang)}")
            # print(f"len of _y:{len(_y)}")

            # print(f"_x_lang:{_x_lang}")

            X_POSES.append(_x)
            X_LANG.append(_x_lang)
            Y_ACTION.append(_y)

        epi_len = len(X_POSES)
        # X = np.array(X)
        # Y = np.array(Y)
        # print(f"X shape: {X.shape}, Y shape: {Y.shape}")
        # (N, 12), (N, 28), int
        return (X_POSES, X_LANG, Y_ACTION, epi_len)

    def one_hot_action(self, action):
        return self.OneHotActionEncoder.encode(action)

    def _pose_dict_to_list(
        self,
        pose_dict: dict,
        withPosition: bool = True,
    ) -> list:

        return (
            [
                pose_dict["position"]["x"],
                pose_dict["position"]["y"],
                pose_dict["position"]["z"],
                pose_dict["rotation"]["x"],
                pose_dict["rotation"]["y"],
                pose_dict["rotation"]["z"],
                pose_dict["rotation"]["w"],
            ]
            if withPosition
            else [
                pose_dict["rotation"]["x"],
                pose_dict["rotation"]["y"],
                pose_dict["rotation"]["z"],
                pose_dict["rotation"]["w"],
            ]
        )
