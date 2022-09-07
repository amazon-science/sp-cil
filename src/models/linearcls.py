# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F


class LinearCls(nn.Module):
    def __init__(self, feat_size, n_class, norm=0):
        super(LinearCls, self).__init__()
        self.feat_size = feat_size
        self.n_class = n_class
        self.norm = norm
        if self.norm > 0:
            self.linear = nn.Linear(feat_size, n_class, bias=False)
        else:
            self.linear = nn.Linear(feat_size, n_class)

    def forward(self, x):
        if self.norm > 0:
            weight = F.normalize(self.linear.weight, p=2, dim=1) * self.norm
            return torch.mm(x, weight.t())

        return self.linear(x)


def linearcls(**kwargs):
    model = LinearCls(**kwargs)
    return model
