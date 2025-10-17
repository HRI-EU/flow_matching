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
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import os

# 替换为你的数据文件夹路径
data_from_folder = r"C:\Users\Public\ChaosProjects\flow_matching\data\unity_data"
# 替换为你的输出文件夹路径
data_output_folder = r"C:\Users\Public\ChaosProjects\flow_matching\data\unity_world_pose_data"

# 确保输出文件夹存在
os.makedirs(data_output_folder, exist_ok=True)

# 处理每个jsonl文件
for filename in os.listdir(data_from_folder):
    if filename.endswith(".jsonl"):
        input_path = os.path.join(data_from_folder, filename)
        output_path = os.path.join(data_output_folder, filename)
        dict_list = []
        with open(input_path, "r") as f_in:
            for json_line in f_in:
                try:
                    data_dict = json.loads(json_line.strip())
                    # 这里的 "Head" 数据是world 坐标系下的
                    # 这里的 "Eye", "LeftHand", "RightHand", "TargetObject" 数据是相对于"Head" 的 local坐标系下的
                    # parent_pos = [v for k, v in data_dict["Head"]["position"].items()]  # x, y, z
                    parent_pos = np.array([0,0,0])  # 这里的Head位置是就是world坐标系下的原点
                    parent_rot_quat = [v for k, v in data_dict["Head"]["rotation"].items()]  # x, y, z, w
                    parent_pos = np.array(parent_pos)
                    parent_rot_quat = np.array(parent_rot_quat)  # xyzw 顺序

                    for key in ["Eye", "LeftHand", "RightHand", "TargetObject"]:
                        if key in data_dict:
                            data = data_dict[key]
                            child_local_pos = [v for k, v in data["position"].items()]  # x, y, z
                            child_local_rot_quat = [v for k, v in data["rotation"].items()]
                            child_local_pos = np.array(child_local_pos)
                            child_local_rot_quat = np.array(child_local_rot_quat)  # xyzw 顺序
                            # 注意: scipy中Rotation默认四元数顺序为 (x, y, z, w)
                            parent_rot = R.from_quat(parent_rot_quat)
                            child_local_rot = R.from_quat(child_local_rot_quat)
                            # 计算Child的World位置
                            child_world_pos = parent_pos + parent_rot.apply(child_local_pos)
                            # 计算Child的World旋转
                            child_world_rot = parent_rot * child_local_rot
                            child_world_quat = child_world_rot.as_quat()
                            # 将结果转换为字典
                            child_world_pos_dict = {
                                "x": float(child_world_pos[0]),
                                "y": float(child_world_pos[1]),
                                "z": float(child_world_pos[2])
                            }
                            child_world_rot_dict = {
                                "x": float(child_world_quat[0]),
                                "y": float(child_world_quat[1]),
                                "z": float(child_world_quat[2]),
                                "w": float(child_world_quat[3])
                            }
                            # 更新字典
                            data_dict[key]["position"] = child_world_pos_dict
                            data_dict[key]["rotation"] = child_world_rot_dict
                    dict_list.append(data_dict)
                except Exception as e:
                    print(f"Error processing line in {filename}: {e}")
        # 写入新的jsonl文件
        with open(output_path, "w") as f_out:
            for item in dict_list:
                f_out.write(json.dumps(item) + "\n")
