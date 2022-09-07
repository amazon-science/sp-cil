# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import torch.nn.functional as F


class DeltaCls(nn.Module):
    def __init__(self, feat_size, n_bclass, n_nclass, norm=-1):
        super(DeltaCls, self).__init__()
        self.feat_size = feat_size
        self.n_bclass = n_bclass
        self.n_nclass = n_nclass
        self.norm = norm
        if self.norm > 0:
            self.linear_base = nn.Linear(feat_size, n_bclass, bias=False)
            self.linear_novel = nn.Linear(feat_size, n_nclass, bias=False)
        else:
            self.linear_base = nn.Linear(feat_size, n_bclass)
            self.linear_novel = nn.Linear(feat_size, n_nclass)

        self.delta_base = nn.Linear(feat_size, n_bclass, bias=False)
        self.delta_novel = nn.Linear(feat_size, n_nclass, bias=False)

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

        if route:
            route_out = torch.cat(
                (z_b.max(dim=1)[0].unsqueeze(-1), z_n.max(dim=1)[0].unsqueeze(-1)),
                dim=1,
            )
            return torch.cat([z_b, z_n], dim=1), route_out
        else:
            return torch.cat([z_b, z_n], dim=1)

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


class DeltaClsN(nn.Module):
    def __init__(self, feat_size, n_classes, norm=-1):
        super(DeltaClsN, self).__init__()
        self.feat_size = feat_size
        self.n_classes = n_classes
        assert norm > 0, "Not implemented for norm = 0 and with bias"
        self.norm = norm
        self.linear = nn.ModuleList()
        for n_cls in n_classes:
            self.linear.append(nn.Linear(feat_size, n_cls, bias=False))

        self.delta = nn.Linear(feat_size * len(n_classes), sum(n_classes), bias=False)

        for name, param in self.named_parameters():
            if "linear" in name:
                param.requires_grad = False
        with torch.no_grad():
            self.delta.weight *= 1 - self.block_diag_indicator([l.weight for l in self.linear])

    @classmethod
    def block_diag_indicator(cls, blocks):
        mask0 = torch.tensor(sum([[i] * x.shape[0] for i, x in enumerate(blocks)], [])).to(blocks[0])
        mask1 = torch.tensor(sum([[i] * x.shape[1] for i, x in enumerate(blocks)], [])).to(blocks[0])
        mask = (mask0[:, None] == mask1[None, :]).to(blocks[0])
        return mask

    def forward(self, *xs, route=False):
        assert len(xs) == len(self.linear)
        # block-diagonal entries
        weights = [F.normalize(l.weight, p=2, dim=1) * self.norm for l in self.linear]
        z_0s = [torch.mm(x, weight.t()) for x, weight in zip(xs, weights)]
        z_0 = torch.cat(z_0s, dim=1)

        # off-block-diagonal entries
        mask = self.block_diag_indicator(weights)
        z_delta = torch.mm(torch.cat(xs, dim=1), (self.delta.weight * (1 - mask)).t())

        z = z_0 + z_delta
        if route:
            start = 0
            route_out = []
            for n_cls in self.n_classes:
                end = start + n_cls
                route_out.append(z[:, start:end].max(dim=1)[0].unsqueeze(-1))
                start = end
            route_out = torch.cat(route_out, dim=1)
            return z, route_out
        else:
            return z

    def load_state_dict2(self, *state_dicts):
        own_state = self.state_dict()
        for idx, state_dict in enumerate(state_dicts):
            # load the idx-th classifier
            for name, param in state_dict.items():
                name = name.replace("linear", "linear.{}".format(idx))
                if name in own_state:
                    if "bias" in name:
                        own_state[name].copy_(param[: self.n_classes[idx]])
                    else:
                        own_state[name].copy_(param[: self.n_classes[idx], :])
                else:
                    print(name)


def deltacls(**kwargs):
    model = DeltaCls(**kwargs)
    return model


def deltaclsn(**kwargs):
    model = DeltaClsN(**kwargs)
    return model
