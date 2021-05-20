import math

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.batchnorm import _BatchNorm


class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Lateral layers
        self.latlayer_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_3 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        self.logits = nn.Sequential(
            nn.Linear(2048, nattr),
            nn.BatchNorm1d(nattr)
        )
        self.logits1 = nn.Sequential(
            nn.Linear(768, nattr),
            nn.BatchNorm1d(nattr)
        )
        self.logits2 = nn.Sequential(
            nn.Linear(512, nattr),
            nn.BatchNorm1d(nattr)
        )
        self.logits3 = nn.Sequential(
            nn.Linear(256, nattr),
            nn.BatchNorm1d(nattr)
        )

    def fresh_params(self):
        return self.parameters()

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        up_feat = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
        return torch.cat([up_feat,y], 1)

    def forward(self, features):
        feature1,feature2,feature3 = features
        # feat1 = self.avg_pool(feature1).view(feature.size(0), -1)
        # feat2 = self.avg_pool(feature2).view(feature.size(0), -1)
        feat3 = self.avg_pool(feature3).view(feature3.size(0), -1)
        logit_main = self.logits(feat3)

        fusion_3 = self.latlayer_3(feature3)
        fusion_2 = self._upsample_add(fusion_3, self.latlayer_2(feature2))
        fusion_1 = self._upsample_add(fusion_2, self.latlayer_1(feature1))

        feat1 = self.avg_pool(fusion_1).view(feature1.size(0), -1)
        feat2 = self.avg_pool(fusion_2).view(feature2.size(0), -1)
        feat3 = self.avg_pool(fusion_3).view(feature3.size(0), -1)

        logits1 = self.logits1(feat1)
        logits2 = self.logits2(feat2)
        logits3 = self.logits3(feat3)



        return logits1,logits2,logits3, logit_main


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
        feat_map1,feat_map2,feat_map3 = self.backbone(x)
        logits = self.classifier((feat_map1,feat_map2,feat_map3))
        return logits
