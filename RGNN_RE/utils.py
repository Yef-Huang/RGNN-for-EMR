import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CE_Label_Smooth_Loss(nn.Module):
    def __init__(self, epsilon=0.1):
        super(CE_Label_Smooth_Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
                 self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
       refer to: https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py#L18
    """
    def __init__(self, smoothing=0.1):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# 设置随机种子
def seed_all(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 情绪标签平滑
def EmotionDL(label, noise=0.6):
    if label == 0:
        new_label = [1 - 2 * noise / 3, 2 * noise / 3, 0]
    elif label == 1:
        new_label = [noise / 3, 1 - 2 * noise / 3, noise / 3]
    elif label == 2:
        new_label = [0, 2 * noise / 3, 1 - 2 * noise / 3]
    return new_label


# 归一化
def extend_normal(sample):
    for i in range(len(sample)):
        features_min = np.min(sample[i])
        features_max = np.max(sample[i])
        sample[i] = (sample[i] - features_min) / (features_max - features_min)
    return sample


def get_ASM_electrode_indices(ch_names, EEG_asymmetric_pairs):
    N_pairs = len(EEG_asymmetric_pairs)  # 14对
    electrode_indices = np.zeros((N_pairs, 2))  # 14*2
    for pair, pair_cnt in zip(EEG_asymmetric_pairs, range(N_pairs)):  # 14*2，给每对导编号
        electrode_left = pair[0]
        electrode_right = pair[1]
        index_left = ch_names.index(electrode_left)
        index_right = ch_names.index(electrode_right)
        electrode_indices[pair_cnt, 0] = index_left
        electrode_indices[pair_cnt, 1] = index_right
    # 找出电导对在EEG_channels_geneva的索引
    electrode_indices = electrode_indices.astype(np.uint8)
    return electrode_indices


def distance_square(location, i, j):
    return (location[i][0] - location[j][0]) * (location[i][0] - location[j][0]) + (location[i][1] - location[j][1]) * (
                location[i][1] - location[j][1])


# 生成邻接矩阵
def get_adj(num_node, pos, global_connections=None):
    A = np.zeros((num_node, num_node))
    for i in range(num_node):
        for j in range(num_node):
            k = distance_square(pos, i, j)  # 计算距离的平方
            np.seterr(divide='ignore', invalid='ignore')  # 忽略除0警告
            A[i][j] = min(1, 5 / k)
    if global_connections is not None:
        for pair in global_connections:
            A[pair[0]][pair[1]] = A[pair[1]][pair[0]] = A[pair[0]][pair[1]] - 1
    return A


# 获取边的信息
def get_edge_info(A):
    node_out = []
    node_in = []
    num_node = len(A)
    for i in range(num_node):
        for j in range(num_node - i):
            node_out.append(i)
            node_in.append(num_node - j - 1)
    node_in = np.array(node_in)
    node_out = np.array(node_out)
    edge_index = np.vstack((node_out, node_in))
    edge_index = torch.from_numpy(edge_index)
    edge_index = edge_index.long()

    edge_weight = []
    num_edge = np.size(edge_index, 1)
    for i in range(num_edge):
        edge_weight.append(A[node_out[i]][node_in[i]])
    edge_weight = np.array(edge_weight)
    edge_weight = torch.from_numpy(edge_weight)
    edge_weight = torch.abs(edge_weight)
    edge_weight = edge_weight.float()
    return edge_index, edge_weight


