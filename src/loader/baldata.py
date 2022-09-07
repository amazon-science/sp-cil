# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from collections import Counter
import logging
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from .img_flist import ImageFilelist

logger = logging.getLogger("mylogger")


class BalancedSampler(Sampler):
    def __init__(self, dataset, tgt_transform=None):

        self.idx = dataset.indices
        class_ids = np.asarray(dataset.dataset.data["ClsID"])[self.idx]
        if tgt_transform is not None:
            class_ids = list(map(tgt_transform, class_ids))

        self.n_samples = len(class_ids)

        # compute class frequencies and set them as sampling weights
        counts = Counter(class_ids)
        get_freq = lambda x: 1.0 / counts[x]
        self.weights = torch.DoubleTensor(list(map(get_freq, class_ids)))

    def __iter__(self):
        sampled_idx = torch.multinomial(self.weights, self.n_samples, replacement=True)
        return (i for i in sampled_idx)

    def __len__(self):
        return self.n_samples


def get_nshot_data(dataset, nshot=0):
    class_ids = np.array(dataset.data["ClsID"])
    classes = np.unique(class_ids)
    if nshot > 0:
        sampled_idx = []
        for i in classes:
            idx = np.where(class_ids == i)[0]
            selected = torch.randperm(len(idx))[:nshot]
            if nshot > 1:
                sampled_idx.extend(idx[selected].tolist())
            else:
                sampled_idx.append(idx[selected])

    else:
        sampled_idx = list(range(len(class_ids)))

    return sampled_idx


def BalancedDataLoader(cfg, splits, batch_size):

    data_loader = dict()

    tgt_transform = lambda x: int(x >= 800)

    for split in splits:

        if split == "train":
            # data augmentation
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            load_sp_lbl = cfg.get("load_sp_lbl", False)
            print("load_sp_lbl", load_sp_lbl)

        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            load_sp_lbl = False

        dataset = ImageFilelist(cfg["root_dir"], cfg[split], transform, load_sp_lbl=load_sp_lbl)
        if split == "train":
            dataset = data.Subset(dataset, get_nshot_data(dataset, nshot=cfg.get("n_shot", 0)))
            sampler = BalancedSampler(dataset, tgt_transform)
            data_loader[split] = data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                sampler=sampler,
                pin_memory=True,
                num_workers=cfg["n_workers"],
            )
        else:
            data_loader[split] = data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=cfg["n_workers"],
            )
        print("{split}: {size}".format(split=split, size=len(dataset)))
        logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    print("Building data loader with {} workers.".format(cfg["n_workers"]))
    logger.info("Building data loader with {} workers.".format(cfg["n_workers"]))

    return data_loader
