from __future__ import print_function, division
from abc import ABC
import torch
import torch.nn as nn


class ConvLayer(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU() if relu_type == 'leaky' else nn.ReLU()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ChannelAttention(nn.Module, ABC):
    def __init__(self, in_channels, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Linear(in_channels * 2, in_channels * 2 * ratio)
        self.fc2 = nn.Linear(in_channels * 2 * ratio, in_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.squeeze(self.avg_pool(x))
        x_max = torch.squeeze(self.max_pool(x))
        if x.shape[0] == 1:
            x_avg = torch.unsqueeze(x_avg, dim=0)
            x_max = torch.unsqueeze(x_max, dim=0)
        x = torch.cat((x_avg, x_max), dim=1)
        x = self.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return x


class SpatialAttention(nn.Module, ABC):
    def __init__(self, kernel_size=7, ratio=2):
        super(SpatialAttention, self).__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(2, 2 * ratio, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv3d(2 * ratio, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        in_channels = x.shape[1]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        x_tmp = x
        for i in range(in_channels - 1):
            x = torch.cat((x, x_tmp), dim=1)
        return x


class Attention(nn.Module, ABC):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        ca = self.ca(x)
        sa = self.sa(x)
        attention = torch.zeros_like(sa)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                attention[i][j] = sa[i][j] * ca[i][j]
        # for i in range(attention.shape[1]):
        #     attention[:, i] = attention[:, i] * ca[:, i]
        anti_attention = torch.ones(attention.shape).cuda() - attention
        return attention, anti_attention


class Interaction(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, dropout_rate, pooling_kernel, pooling_stride, pooling_padding, factor):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels * factor, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pooling = nn.MaxPool3d(pooling_kernel, pooling_stride, pooling_padding)
        self.attention_c = Attention(in_channels)
        self.attention_r = Attention(in_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_c, x_r, x_ori=None):
        x = torch.cat((x_c, x_r), dim=1)
        if x_ori is not None:
            x = torch.cat((x_ori, x), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        a_c, _ = self.attention_c(x_c)
        a_r, _ = self.attention_r(x_r)
        x = self.relu(self.conv(x_c * a_c + x_r * a_r)) + x
        x = self.pooling(x)

        x = self.dropout(x)
        return x


class _CNN(nn.Module, ABC):
    def __init__(self, fil_num, drop_rate):
        super(_CNN, self).__init__()

        self.block_c_1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block_c_2 = ConvLayer(fil_num, 2 * fil_num, 0.1, (5, 1, 2), (3, 2, 0))
        self.block_c_3 = ConvLayer(4 * fil_num, 4 * fil_num, 0.1, (5, 1, 0), (3, 2, 0))
        self.block_c_4 = ConvLayer(4 * fil_num, 4 * fil_num, 0.1, (3, 1, 1), (3, 1, 0))
        self.block_c_5 = ConvLayer(8 * fil_num, 8 * fil_num, 0.1, (3, 1, 0), (3, 1, 0))
        self.block_c_6 = ConvLayer(8 * fil_num, 8 * fil_num, 0.1, (3, 1, 1), (1, 1, 0))

        self.block_r_1 = ConvLayer(1, fil_num, 0.1, (7, 2, 0), (3, 2, 0))
        self.block_r_2 = ConvLayer(fil_num, 2 * fil_num, 0.1, (5, 1, 2), (3, 2, 0))
        self.block_r_3 = ConvLayer(4 * fil_num, 4 * fil_num, 0.1, (5, 1, 0), (3, 2, 0))
        self.block_r_4 = ConvLayer(4 * fil_num, 4 * fil_num, 0.1, (3, 1, 1), (3, 1, 0))
        self.block_r_5 = ConvLayer(8 * fil_num, 8 * fil_num, 0.1, (3, 1, 0), (3, 1, 0))
        self.block_r_6 = ConvLayer(8 * fil_num, 8 * fil_num, 0.1, (3, 1, 1), (1, 1, 0))

        self.attention_c_1 = Attention(fil_num)
        self.attention_r_1 = Attention(fil_num)
        self.block_i_1 = Interaction(fil_num, 2 * fil_num, 0.1, 3, 2, 0, 2)
        self.block_i_2 = ConvLayer(2 * fil_num, 4 * fil_num, 0.1, (5, 1, 0), (3, 2, 0))
        self.attention_c_2 = Attention(4 * fil_num)
        self.attention_r_2 = Attention(4 * fil_num)
        self.block_i_3 = Interaction(4 * fil_num, 4 * fil_num, 0.1, 3, 1, 0, 3)
        self.block_i_4 = ConvLayer(4 * fil_num, 8 * fil_num, 0.1, (3, 1, 0), (3, 1, 0))
        self.attention_c_3 = Attention(8 * fil_num)
        self.attention_r_3 = Attention(8 * fil_num)
        self.block_i_5 = Interaction(8 * fil_num, 8 * fil_num, 0.1, 1, 1, 0, 3)

        self.dense_c = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256 * fil_num, 32)
        )
        self.dense_r = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(256 * fil_num, 32)
        )
        self.classify = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 2), )
        self.regress = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        per_loss = torch.zeros(x.shape[0]).cuda()
        x_c = self.block_c_1(x)
        x_r = self.block_r_1(x)
        a_c, anti_a_c = self.attention_c_1(x_c)
        a_r, anti_a_r = self.attention_r_1(x_r)
        per_loss += self.perceptual_loss(a_c * x_c, a_r * x_r)

        x = self.block_i_1(a_c * x_c, a_r * x_r)
        x_c = self.block_c_2(anti_a_c * x_c)
        x_r = self.block_r_2(anti_a_r * x_r)

        x_c = self.block_c_3(torch.cat((x, x_c), dim=1))
        x_r = self.block_r_3(torch.cat((x, x_r), dim=1))
        x = self.block_i_2(x)
        a_c, anti_a_c = self.attention_c_2(x_c)
        a_r, anti_a_r = self.attention_r_2(x_r)
        per_loss += self.perceptual_loss(a_c * x_c, a_r * x_r)

        x = self.block_i_3(a_c * x_c, a_r * x_r, x)
        x_c = self.block_c_4(anti_a_c * x_c)
        x_r = self.block_r_4(anti_a_r * x_r)

        x_c = self.block_c_5(torch.cat((x_c, x), dim=1))
        x_r = self.block_r_5(torch.cat((x_r, x), dim=1))
        x = self.block_i_4(x)
        a_c, anti_a_c = self.attention_c_3(x_c)
        a_r, anti_a_r = self.attention_r_3(x_r)
        per_loss += self.perceptual_loss(a_c * x_c, a_r * x_r)

        x = self.block_i_5(a_c * x_c, a_r * x_r, x)
        x_c = self.block_c_6(anti_a_c * x_c)
        x_r = self.block_r_6(anti_a_r * x_r)

        x_c = torch.cat((x, x_c), dim=1)
        x_r = torch.cat((x, x_r), dim=1)

        batch_size = x.shape[0]
        x_c = x_c.view(batch_size, -1)
        x_r = x_r.view(batch_size, -1)
        x_c = self.dense_c(x_c)
        x_r = self.dense_r(x_r)
        output_c = self.classify(x_c)
        output_r = self.regress(x_r)
        return output_c, output_r, per_loss

    @staticmethod
    def perceptual_loss(x1, x2):
        avg1 = torch.mean(x1, dim=1, keepdim=True)
        max1, _ = torch.max(x1, dim=1, keepdim=True)
        x1 = torch.cat((avg1, max1), dim=1)
        avg2 = torch.mean(x2, dim=1, keepdim=True)
        max2, _ = torch.max(x2, dim=1, keepdim=True)
        x2 = torch.cat((avg2, max2), dim=1)

        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        per_loss = torch.zeros(x1.shape[0]).cuda()
        loss = torch.nn.L1Loss()
        for i in range(x1.shape[0]):
            per_loss[i] += loss(x1[i], x2[i])
        per_loss /= x1.shape[0]
        return per_loss
