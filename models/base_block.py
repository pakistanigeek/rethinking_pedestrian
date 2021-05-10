import math

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch
from torch.nn import functional as F

class ChannelAttn(nn.Module):
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels%reduction_rate == 0
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_rate, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels // reduction_rate, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return torch.sigmoid(x)


class SpatialTransformBlock(nn.Module):
    def __init__(self, num_classes, pooling_size, channels):
        super(SpatialTransformBlock, self).__init__()
        self.num_classes = num_classes
        self.spatial = pooling_size

        self.global_pool = nn.AvgPool2d((pooling_size[0], pooling_size[1]), stride=1, padding=0, ceil_mode=True, count_include_pad=True)

        self.gap_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        self.att_list = nn.ModuleList()
        self.stn_list = nn.ModuleList()
        for i in range(self.num_classes):
            self.gap_list.append(nn.AvgPool2d((pooling_size[0], pooling_size[1]), stride=1, padding=0, ceil_mode=True, count_include_pad=True))
            self.fc_list.append(nn.Linear(channels, 1))
            self.att_list.append(ChannelAttn(channels))
            self.stn_list.append(nn.Linear(channels, 4))

    def stn(self, x, theta):
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, padding_mode='border')
        # return x.cuda()
        return x

    def transform_theta(self, theta_i, region_idx):
        theta = torch.zeros(theta_i.size(0), 2, 3)
        theta[:,0,0] = torch.sigmoid(theta_i[:,0])
        theta[:,1,1] = torch.sigmoid(theta_i[:,1])
        theta[:,0,2] = torch.tanh(theta_i[:,2])
        theta[:,1,2] = torch.tanh(theta_i[:,3])
        # theta = theta.cuda()
        return theta

    def forward(self, features):
        pred_list = []
        bs = features.size(0)
        for i in range(self.num_classes):
            stn_feature = features * self.att_list[i](features) + features

            theta_i = self.stn_list[i](F.avg_pool2d(stn_feature, stn_feature.size()[2:]).view(bs,-1)).view(-1,4)
            theta_i = self.transform_theta(theta_i, i)

            sub_feature = self.stn(stn_feature, theta_i)
            pred = self.gap_list[i](sub_feature).view(bs,-1)
            pred = self.fc_list[i](pred)
            pred_list.append(pred)
        pred = torch.cat(pred_list, 1)
        return pred


class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        self.logits = nn.Sequential(
            nn.Linear(2048, nattr),
            nn.BatchNorm1d(nattr)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.st_1 = SpatialTransformBlock(nattr, (64,48), 256 * 4)
        self.st_2 = SpatialTransformBlock(nattr, (32, 24), 256 * 3)
        self.st_3 = SpatialTransformBlock(nattr, (16,12), 256 * 2)
        self.st_4 = SpatialTransformBlock(nattr, (8,6), 256)

        # Lateral layers
        self.latlayer_1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_3 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_4 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

    def fresh_params(self):
        return self.parameters()

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        up_feat = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
        return torch.cat([up_feat,y], 1)

    def forward(self, feature):
        feature1,feature2,feature3,feature4 = feature
        feat = self.avg_pool(feature4).view(feature4.size(0), -1)
        pred_main = self.logits(feat)

        fusion_4 = self.latlayer_4(feature4)
        fusion_3 = self._upsample_add(fusion_4, self.latlayer_3(feature3))
        fusion_2 = self._upsample_add(fusion_3, self.latlayer_2(feature2))
        fusion_1 = self._upsample_add(fusion_2, self.latlayer_1(feature1))

        pred_1 = self.st_1(fusion_1)
        pred_2 = self.st_2(fusion_2)
        pred_3 = self.st_3(fusion_3)
        pred_4 = self.st_4(fusion_4)
        return pred_main,pred_1,pred_2,pred_3,pred_4


def initialize_weights(module):
    for m in module.children():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, _BatchNorm):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)


class FeatClassifier(nn.Module):

    def __init__(self, backbone, classifier):
        super(FeatClassifier, self).__init__()

        self.backbone = backbone
        self.classifier = classifier

    def fresh_params(self):
        params = self.classifier.fresh_params()
        return params

    def finetune_params(self):
        return self.backbone.parameters()

    def forward(self, x, label=None):
        feat_map1,feat_map2,feat_map3,feat_map4 = self.backbone(x[0])
        logits1, logits2, logits3, logits4, logit5 = self.classifier((feat_map1, feat_map2, feat_map3,feat_map4))
        return logits1, logits2, logits3, logits4, logit5
