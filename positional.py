import math  # 导入数学库
import torch  # 导入PyTorch
import torch.nn as nn  # 导入PyTorch的神经网络模块
from einops import rearrange, repeat  # 从einops库中导入rearrange和repeat函数





def PositionalEncoder(image_shape,num_frequency_bands,max_frequencies=None):  # 定义位置编码器函数
    
    *spatial_shape, _ = image_shape  # 获取图像形状
    
    coords = [ torch.linspace(-1, 1, steps=s) for s in spatial_shape ]  # 生成坐标
    pos = torch.stack(torch.meshgrid(*coords), dim=len(spatial_shape))  # 生成位置编码
    
    encodings = []  # 初始化编码列表
    if max_frequencies is None:  # 如果最大频率未指定
        max_frequencies = pos.shape[:-1]  # 使用位置编码的形状

    frequencies = [ torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
                                              for max_freq in max_frequencies ]  # 生成频率
    
    frequency_grids = []  # 初始化频率网格
    for i, frequencies_i in enumerate(frequencies):  # 遍历频率
        frequency_grids.append(pos[..., i:i+1] * frequencies_i[None, ...])  # 生成频率网格

    encodings.extend([torch.sin(math.pi * frequency_grid) for frequency_grid in frequency_grids])  # 添加正弦编码
    encodings.extend([torch.cos(math.pi * frequency_grid) for frequency_grid in frequency_grids])  # 添加余弦编码
    enc = torch.cat(encodings, dim=-1)  # 拼接编码
    enc = rearrange(enc, "... c -> (...) c")  # 重新排列编码

    return enc  # 返回编码






