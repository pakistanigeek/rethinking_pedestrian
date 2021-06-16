import math

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm


class BaseClassifier(nn.Module):
    def __init__(self, nattr):
        super().__init__()
        self.logits1 = nn.Sequential(
            nn.Linear(1024, nattr),
            nn.BatchNorm1d(nattr)
        )
        self.logits2 = nn.Sequential(
            nn.Linear(2048, nattr),
            nn.BatchNorm1d(nattr)
        )

        self.pooling = nn.MaxPool2d((2), stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def fresh_params(self):
        return self.parameters()

    def forward(self, feature):
        feature1, feature2 = feature
        feature1 = self.pooling(feature1)
        feat1 = self.avg_pool(feature1).view(feature1.size(0), -1)
        feat2 = self.avg_pool(feature2).view(feature2.size(0), -1)

        logits1 = self.logits1(feat1)
        logits2 = self.logits2(feat2)
        return logits1, logits2


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
        feat_map1,feat_map2 = self.backbone(x)
        logits = self.classifier((feat_map1,feat_map2))
        return logits
