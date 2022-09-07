# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F


class DeltaPoolCls(nn.Module):
    def __init__(self, feat_size, n_bclass, n_nclass, n_overlap=0, pool="max", norm=-1):
        super(DeltaPoolCls, self).__init__()
        assert n_overlap <= n_bclass and n_overlap <= n_nclass
        self.feat_size = feat_size
        self.n_bclass = n_bclass
        self.n_nclass = n_nclass
        self.n_overlap = n_overlap
        self.norm = norm
        if norm > 0:
            self.linear_base = nn.Linear(feat_size, n_bclass, bias=False)
            self.linear_novel = nn.Linear(feat_size, n_nclass, bias=False)
        else:
            self.linear_base = nn.Linear(feat_size, n_bclass)
            self.linear_novel = nn.Linear(feat_size, n_nclass)

        self.delta_base = nn.Linear(feat_size, n_bclass, bias=False)
        self.delta_novel = nn.Linear(feat_size, n_nclass, bias=False)

        if n_overlap > 0:
            if pool == "avg":
                self.pool = nn.AvgPool2d(kernel_size=(2, 1))
            else:
                self.pool = nn.MaxPool2d(kernel_size=(2, 1))

        for name, param in self.named_parameters():
            if "linear" in name:
                param.requires_grad = False

    def forward(self, x_b, x_n, route=False):
        if self.norm > 0:
            weight_b = F.normalize(self.linear_base.weight, p=2, dim=1) * self.norm
            weight_n = F.normalize(self.linear_novel.weight, p=2, dim=1) * self.norm
            z_b0 = torch.mm(x_b, weight_b.t())
            z_n0 = torch.mm(x_n, weight_n.t())
        else:
            z_b0 = self.linear_base(x_b)
            z_n0 = self.linear_novel(x_n)

        delta_z_b = self.delta_base(x_n)
        delta_z_n = self.delta_novel(x_b)

        z_b = z_b0 + delta_z_b
        z_n = z_n0 + delta_z_n

        if self.n_overlap > 0:
            z_overlap = torch.cat(
                [
                    z_b[:, : self.n_overlap].unsqueeze(1),
                    z_n[:, : self.n_overlap].unsqueeze(1),
                ],
                dim=1,
            )
            z_overlap = self.pool(z_overlap).squeeze()
            z = torch.cat([z_overlap, z_b[:, self.n_overlap :], z_n[:, self.n_overlap :]], dim=1)
        else:
            z = torch.cat([z_b, z_n], dim=1)

        if route:
            route_out = torch.cat(
                (z_b.max(dim=1)[0].unsqueeze(-1), z_n.max(dim=1)[0].unsqueeze(-1)),
                dim=1,
            )
            return z, route_out

        return z

    def load_state_dict2(self, state_dict1, state_dict2):
        own_state = self.state_dict()
        # load the base classifier
        for name, param in state_dict1.items():
            name = name.replace("linear", "linear_base")
            if name in own_state:
                if "bias" in name:
                    own_state[name].copy_(param[: self.n_bclass])
                else:
                    own_state[name].copy_(param[: self.n_bclass, :])
            else:
                print(name)

        # load the novel classifier
        for name, param in state_dict2.items():
            name = name.replace("linear", "linear_novel")
            if name in own_state:
                own_state[name].copy_(param)
            else:
                print(name)


def deltapoolcls(**kwargs):
    model = DeltaPoolCls(**kwargs)
    return model
