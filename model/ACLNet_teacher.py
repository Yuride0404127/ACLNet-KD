import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from einops import rearrange

from backbone.Shunted_Transformer.SSA import shunted_b, shunted_s, shunted_t

stage1_channel = 64
stage2_channel = 128
stage3_channel = 256
stage4_channel = 512


class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)


class SalHead(nn.Module):
    def __init__(self, in_channel):
        super(SalHead, self).__init__()
        self.conv = nn.Sequential(
                nn.Dropout2d(p=0.1),
                nn.Conv2d(in_channel, 1, 1, stride=1, padding=0),
                # nn.Sigmoid()
                )

    def forward(self, x):
        return self.conv(x)


class WF_V2(nn.Module):
    def __init__(self):
        super(WF_V2, self).__init__()
        self.alph = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_q = rearrange(x, 'b c h w -> b c (h w)')
        x_k = rearrange(x, 'b c h w -> b (h w) c')

        x_dot_c = torch.bmm(x_q, x_k)

        x_q_c = torch.norm(x_q, p=2, dim=2).view(b, -1, c).permute(0, 2, 1)
        x_k_c = torch.norm(x_k, p=2, dim=1).view(b, -1, c)
        x_dot_c_ = torch.bmm(x_q_c, x_k_c) + 1e-08
        atten_map_c = torch.div(x_dot_c, x_dot_c_)
        x_v_c = x.view(b, c, -1)
        out_c = torch.bmm(atten_map_c, x_v_c)
        out_c = out_c.view(b, c, h, w)

        out = self.alph * out_c + x

        return out


class teacher_model_v4(nn.Module):
    def __init__(self):
        super(teacher_model_v4, self).__init__()
        # Backbone model
        self.rgb = shunted_b(pretrained=True)
        self.depth = shunted_b(pretrained=True)


        self.Head1 = SalHead(stage1_channel)
        self.Head2 = SalHead(stage2_channel)
        self.Head3 = SalHead(stage3_channel)
        self.Head4 = SalHead(stage4_channel)

        self.upsample32 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.wf_1 = WF_V2()
        self.wf_2 = WF_V2()
        self.wf_3 = WF_V2()
        self.wf_4 = WF_V2()

        self.conv_cross_1 = convbnrelu(stage2_channel, stage1_channel, k=1, s=1, p=0, relu=True)
        self.conv_cross_2 = convbnrelu(stage3_channel, stage2_channel, k=1, s=1, p=0, relu=True)
        self.conv_cross_3 = convbnrelu(stage4_channel, stage3_channel, k=1, s=1, p=0, relu=True)

        self.conv_concat_1 = convbnrelu(2 * stage1_channel, stage1_channel, k=1, s=1, p=0, relu=True)
        self.conv_concat_2 = convbnrelu(2 * stage2_channel, stage2_channel, k=1, s=1, p=0, relu=True)
        self.conv_concat_3 = convbnrelu(2 * stage3_channel, stage3_channel, k=1, s=1, p=0, relu=True)
        self.conv_concat_4 = convbnrelu(2 * stage4_channel, stage4_channel, k=1, s=1, p=0, relu=True)

    def forward(self, x_rgb, x_depth):
        rgb_list = self.rgb(x_rgb)
        rgb_1 = rgb_list[0]
        rgb_2 = rgb_list[1]
        rgb_3 = rgb_list[2]
        rgb_4 = rgb_list[3]

        x_depth = torch.cat([x_depth, x_depth, x_depth], dim=1)
        depth_list = self.depth(x_depth)
        depth_1 = depth_list[0]
        depth_2 = depth_list[1]
        depth_3 = depth_list[2]
        depth_4 = depth_list[3]

        stage1_out = self.conv_concat_1(torch.cat([rgb_1, depth_1], dim=1))
        stage2_out = self.conv_concat_2(torch.cat([rgb_2, depth_2], dim=1))
        stage3_out = self.conv_concat_3(torch.cat([rgb_3, depth_3], dim=1))
        stage4_out = self.conv_concat_4(torch.cat([rgb_4, depth_4], dim=1))

        am_out_4 = self.wf_4(stage4_out)
        am_out_3 = self.wf_3(self.upsample2(self.conv_cross_3(am_out_4)) + stage3_out)
        am_out_2 = self.wf_2(self.upsample2(self.conv_cross_2(am_out_3)) + stage2_out)
        am_out_1 = self.wf_1(self.upsample2(self.conv_cross_1(am_out_2)) + stage1_out)


        fuse_1 = self.upsample4(self.Head1(am_out_1))
        fuse_2 = self.upsample8(self.Head2(am_out_2))
        fuse_3 = self.upsample16(self.Head3(am_out_3))
        fuse_4 = self.upsample32(self.Head4(am_out_4))

        # return fuse_1, fuse_2, fuse_3, fuse_4, am_out_1, am_out_2, am_out_3, am_out_4, \
        #        stage1_out, stage2_out, stage3_out, stage4_out
        return  am_out_1


if __name__ == '__main__':
    input_rgb = torch.randn(2, 3, 256, 256)
    input_depth = torch.randn(2, 1, 256, 256)
    net = teacher_model_v4()
    out = net(input_rgb, input_depth)
    # input = torch.randn(2, 768, 7, 7)
    # net = WF_Attention()
    # out = net(input)
    # print("out", out.shape)
    # print("out1", out[0].shape)
    # print("out2", out[1].shape)
    # print("out3", out[2].shape)
    # print("out4", out[3].shape)
    # print("out5", out[4].shape)
    # print("out6", out[5].shape)
    # print("out7", out[6].shape)
    # print("out8", out[7].shape)
    a = torch.randn(1, 3, 256, 256)
    b = torch.randn(1, 1, 256, 256)
    model = teacher_model_v4()
    from FLOP import CalParams

    CalParams(model, a, b)
    print('Total params % .2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
