#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

# 转换世界坐标系到相机坐标系
# 参数： 旋转 平移矩阵 要转换的参数数组 缩放因子
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    # 世界坐标系到相机坐标系的转换矩阵
    # Rt = [[ R_{3*3}, t_{3*1}]
    #       [ 0      , 1      ]]
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # 相机坐标系到世界坐标系到转换矩阵 Rt^{-1} 实际就是相机的内参矩阵
    C2W = np.linalg.inv(Rt)
    # 相机坐标
    cam_center = C2W[:3, 3]
    # 移动缩放得到新相机中心
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    # 再逆回去就是新的世界坐标系到相机坐标系转换矩阵
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

# 相机视野的最近点 最远点 水平方向与垂直放心的视野范围
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))