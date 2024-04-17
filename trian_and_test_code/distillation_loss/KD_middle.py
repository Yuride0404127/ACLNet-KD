import os
import sys
import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from einops import rearrange

class BCELOSS(nn.Module):
    def __init__(self):
        super(BCELOSS, self).__init__()
        self.nll_lose = nn.BCELoss()

    def forward(self, input_scale, taeget_scale):
        losses = []
        for inputs, targets in zip(input_scale, taeget_scale):
            lossall = self.nll_lose(inputs, targets)
            losses.append(lossall)
        total_loss = sum(losses)
        return total_loss

def hcl(fstudent, fteacher):
    loss_all = 0.0
    B, C, h, w = fstudent.size()
    loss = F.mse_loss(fstudent, fteacher, reduction='mean')
    cnt = 1.0
    tot = 1.0
    for l in [4,2,1]:
        if l >=h:
            continue
        tmpfs = F.adaptive_avg_pool2d(fstudent, (l,l))
        tmpft = F.adaptive_avg_pool2d(fteacher, (l,l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
        tot += cnt
    loss = loss / tot
    loss_all = loss_all + loss
    return loss_all

def dice_loss(pred, mask):
    mask = torch.sigmoid(mask)
    pred = torch.sigmoid(pred)
    intersection = (pred * mask).sum(axis=(2, 3))
    unior = (pred + mask).sum(axis=(2, 3))
    dice = (2 * intersection + 1) / (unior + 1)
    dice = torch.mean(1 - dice)
    return dice

class At_loss(nn.Module):
    """Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks
    via Attention Transfer
    code: https://github.com/szagoruyko/attention-transfer"""
    def __init__(self, p=2):
        super(At_loss, self).__init__()
        self.p = p

    def forward(self, g_s, g_t):
        return self.at_loss(g_s, g_t)

    def at_loss(self, f_s, f_t):
        s_H, t_H = f_s.shape[2], f_t.shape[2]
        if s_H > t_H:
            f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
        elif s_H < t_H:
            f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
        else:
            pass
        return (self.at(f_s) - self.at(f_t)).pow(2).mean()

    def at(self, f):
        return F.normalize(f.pow(self.p).mean(1).view(f.size(0), -1))




class  ATD(nn.Module):
    def __init__(self, channel):
        super(Attention_loss, self).__init__()

        # self.conv1_s = nn.Conv2d(channel, channel, 1)

        self.k = 64
        self.linear_0_s = nn.Conv1d(channel, self.k, 1, bias=False)

        self.linear_1_s = nn.Conv1d(channel, self.k, 1, bias=False)
        self.linear_1_s.weight.data = self.linear_0_s.weight.data

        self.linear_re_s = nn.Conv1d(self.k, channel, 1, bias=False)
        # self.conv2_s = nn.Sequential(nn.Conv2d(channel, channel, 1, bias=False),
        #                            nn.BatchNorm2d(channel, eps=1e-4))

        # self.conv1_t = nn.Conv2d(channel, channel, 1)

        self.k = 64
        self.linear_0_t = nn.Conv1d(channel, self.k, 1, bias=False)

        self.linear_1_t = nn.Conv1d(channel, self.k, 1, bias=False)
        self.linear_1_t.weight.data = self.linear_0_t.weight.data

        self.linear_re_t = nn.Conv1d(self.k, channel, 1, bias=False)

        # self.conv2_t = nn.Sequential(nn.Conv2d(channel, channel, 1, bias=False),
        #                              nn.BatchNorm2d(channel, eps=1e-4))

        self.attention_loss = At_loss()

    def forward(self, student, teacher):
        x_s = student
        # x_s = self.conv1_s(x_s)

        b_s, c_s, h_s, w_s = x_s.size()
        x_s = x_s.view(b_s, c_s, -1)

        softmax_s = self.linear_0_s(x_s)
        softmax_s = F.softmax(softmax_s, dim=-1)
        softmax_s = softmax_s / (1e-9 + softmax_s.sum(dim=1, keepdim=True))
        # print("softmax_s", softmax_s.shape)

        linear_project_s = self.linear_1_s(x_s)
        # print("linear_project_s", linear_project_s.shape)

        muti_x_s = linear_project_s * softmax_s
        at_s = self.linear_re_s(muti_x_s)
        at_s = at_s.view(b_s, c_s, h_s, w_s)
        # at_s = self.conv2_s(at_s)
        at_s = at_s + student

        x_t = teacher
        # x_t = self.conv1_t(x_t)
        b_t, c_t, h_t, w_t = x_t.size()
        x_t = x_t.view(b_t, c_t, -1)

        softmax_t = self.linear_0_t(x_t)
        softmax_t = F.softmax(softmax_t, dim=1)
        softmax_t = softmax_t / (1e-9 + softmax_t.sum(dim=1, keepdim=True))

        linear_project_t = self.linear_1_t(x_t)
        muti_x_t = linear_project_t * softmax_t
        at_t = self.linear_re_s(muti_x_t)
        at_t = at_t.view(b_s, c_s, h_s, w_s)
        # at_t = self.conv2_t(at_t)
        at_t = at_t + teacher

        loss_attention = self.attention_loss(at_s, at_t)


        return loss_attention

class  Graph_inflection(nn.Module):
    def __init__(self, HW, channel, node):
        super(Graph_inflection, self).__init__()
        self.Num_node = node
        self.Num_channel = channel

        self.build_node = nn.Conv1d(HW, self.Num_node, 1)

        self.relu = nn.ReLU(inplace=True)

        self.node_conv = nn.Conv1d(self.Num_node, self.Num_node, 1)
        self.channel_conv = nn.Conv1d(self.Num_channel, self.Num_channel, 1)
        self._init_weight()

    def forward(self, x):
        # x:B C
        B, C, H, W = x.shape
        L = H * W
        # print("L", L)
        x_reshape = x.view(-1, C, L) # B, C, L
        # print("x_reshape", x_reshape.shape)
        x_node = self.build_node(x_reshape.permute(0, 2, 1).contiguous()) #x_node: B, N, C
        # print("x_node", x_node.shape)
        Vertex = self.node_conv(x_node) # Vertex : B N C
        # print("Vertex", Vertex.shape)
        Vertex = Vertex + x_node
        # print("Vertex", Vertex.shape)
        Vertex = self.relu(self.channel_conv(Vertex.permute(0, 2, 1).contiguous()))

        return Vertex

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))

class Graph_loss(nn.Module):
    def __init__(self, HW, channel, node):
        super(Graph_loss, self).__init__()
        self.Graph_student = Graph_inflection(HW, channel, node)
        self.Graph_teacher = Graph_inflection(HW, channel, node)

    def forward(self, student, teacher):
        Out_student = self.Graph_student(student).unsqueeze(0)
        # print("Out_student", Out_student.shape)
        Out_teacher = self.Graph_teacher(teacher).unsqueeze(0)
        # print("Out_teacher", Out_teacher.shape)
        graph_loss = dice_loss(Out_student, Out_teacher)

        return graph_loss



class GAP_conv_bn_relu(nn.Module):
    def __init__(self, in_channels, pool_size, rate):
        super(GAP_conv_bn_relu, self).__init__()
        self.AAP = nn.AdaptiveAvgPool2d(pool_size)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=rate)
        self.bn = nn.BatchNorm2d(in_channels, momentum=.95)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.AAP(x)
        x = self.conv(x)
        x = self.relu(self.bn(x))

        return x


class MGMD(nn.Module):
    def __init__(self, in_channel, pool_size):
        super(MGMD, self).__init__()

        self.conv_s_rate1 = GAP_conv_bn_relu(in_channel, pool_size, 1)
        self.conv_s_rate3 = GAP_conv_bn_relu(in_channel, pool_size, 3)
        self.conv_s_rate5 = GAP_conv_bn_relu(in_channel, pool_size, 5)

        self.conv_t_rate1 = GAP_conv_bn_relu(in_channel, pool_size, 1)
        self.conv_t_rate3 = GAP_conv_bn_relu(in_channel, pool_size, 3)
        self.conv_t_rate5 = GAP_conv_bn_relu(in_channel, pool_size, 5)
        # self.sk3to1_t = SKConv_3to1(in_channel, pool_size, 3, 8, 2)
        # self.sk3to1_s = SKConv_3to1(in_channel, pool_size, 3, 8, 2)
        self.graph_loss_1 = Graph_loss(pool_size * pool_size, in_channel, pool_size)
        self.graph_loss_2 = Graph_loss(pool_size * pool_size, in_channel, pool_size)
        self.graph_loss_3 = Graph_loss(pool_size * pool_size, in_channel, pool_size)

    def forward(self, student, teacher):
        x_s_r1 = self.conv_s_rate1(student)
        # print("x_s_r1", x_s_r1.shape)
        x_s_r3 = self.conv_s_rate3(student)
        # print("x_s_r3", x_s_r3.shape)
        x_s_r5 = self.conv_s_rate5(student)
        # print("x_s_r5", x_s_r5.shape)
        # sk_out_s = self.sk3to1_s(x_s_r1, x_s_r3, x_s_r5)

        x_t_r1 = self.conv_t_rate1(teacher)
        x_t_r3 = self.conv_t_rate3(teacher)
        x_t_r5 = self.conv_t_rate5(teacher)
        # sk_out_t = self.sk3to1_t(x_t_r1, x_t_r3, x_t_r5)

        loss_ppa1 = self.graph_loss_1(x_s_r1, x_t_r1)
        loss_ppa2 = self.graph_loss_2(x_s_r3, x_t_r3)
        loss_ppa3 = self.graph_loss_3(x_s_r5, x_t_r5)

        # loss_ppa = hcl(sk_out_s, sk_out_t)
        loss_ppa = loss_ppa1 + loss_ppa2 + loss_ppa3

        return loss_ppa


class CLD(nn.Module):
    def __init__(self, channels, tau = 0.5):
        super(CLD, self).__init__()
        self.tau = tau
        self.conv_z1 = nn.Conv2d(channels, 1, kernel_size=1, stride=1)
        self.conv_z2 = nn.Conv2d(channels, 1, kernel_size=1, stride=1)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
        z1_reshped= rearrange(z1, 'b c h w -> b c (h w)')
        z2_reshped = rearrange(z2, 'b c h w -> b c (h w)')
        z1_c = torch.mean(z1_reshped, dim=-1)
        z2_c = torch.mean(z2_reshped, dim=-1)
        z1_hw = self.conv_z1(z1)
        z1_hw = rearrange(z1_hw, 'b c h w -> b (c h w)')
        z2_hw = self.conv_z1(z2)
        z2_hw = rearrange(z2_hw, 'b c h w -> b (c h w)')

        if batch_size == 0:
            l1 = self.semi_loss(z1_c, z2_c)
            l2 = self.semi_loss(z2_c, z1_c)
            l3 = self.semi_loss(z1_hw, z2_hw)
            l4 = self.semi_loss(z2_hw, z1_hw)
        else:
            l1 = self.batched_semi_loss(z1_c, z2_c, batch_size)
            l2 = self.batched_semi_loss(z2_c, z1_c, batch_size)
            l3 = self.batched_semi_loss(z1_hw, z2_hw, batch_size)
            l4 = self.batched_semi_loss(z2_hw, z1_hw, batch_size)

        ret = (l1 + l2 + l3 + l4) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

# if __name__ == '__main__':
#     teacher = torch.randn(6, 512, 8, 8)
#     student = torch.randn(6, 512, 8, 8)
#     loss1 = Graph_loss(8*8, 512, 8)
#     loss2 = Attention_loss(512)
#     loss3 = PPA_loss(512, 8)
#     SKConv1 = SKConv(512, 8, 3, 8, 2)
#     SKConv2 = SKConv_3to1(512, 8, 3, 8, 2)
#     # teacher = torch.sigmoid(teacher)
#     # student = torch.sigmoid(student)
#     out1 = loss2(student, teacher)
#     print("out1", out1)
#     # print("out1", out1)