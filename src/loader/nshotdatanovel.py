# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import logging
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate

from .img_flist import ImageFilelist

logger = logging.getLogger("mylogger")


def get_collate_func(rot):
    if rot:
        assert not rot, "Rotation no longer supported"
        return None
    else:
        return default_collate


def get_nshot_data(dataset, nshot=0, nshotclass=0, seed=None):
    class_ids = np.array(dataset.data["ClsID"])
    classes = np.unique(class_ids)
    seed = 2147483647 if seed is None else 65535 * seed
    if nshot > 0:
        sampled_idx = []
        for i in classes:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(seed + i)
            idx = np.where(class_ids == i)[0]
            if i < nshotclass:
                selected = torch.randperm(len(idx), generator=g_cpu)[:nshot]
            else:
                selected = torch.randperm(len(idx), generator=g_cpu)
            if nshot > 1:
                sampled_idx.extend(idx[selected].tolist())
            else:
                sampled_idx.append(idx[selected])

    else:
        sampled_idx = list(range(len(class_ids)))

    return sampled_idx


def NshotNovelDataLoader(cfg, splits, batch_size, seed=None):

    data_loader = dict()

    tgt_transform = lambda x: x + cfg.get("lbl_offset", 0)
    collate_func = get_collate_func(cfg.get("rot", False))

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
            shuffle = True
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
            shuffle = False
            load_sp_lbl = False

        dataset = ImageFilelist(
            cfg["root_dir"],
            cfg[split],
            transform,
            tgt_transform,
            load_sp_lbl=load_sp_lbl,
        )
        if split == "train":
            dataset = data.Subset(
                dataset,
                get_nshot_data(
                    dataset,
                    nshot=cfg["n_shot"],
                    nshotclass=cfg["nshotclass"],
                    seed=seed,
                ),
            )
        data_loader[split] = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            collate_fn=collate_func,
            pin_memory=True,
            num_workers=cfg["n_workers"],
        )
        print("{split}: {size}".format(split=split, size=len(dataset)))
        logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    print("Building data loader with {} workers.".format(cfg["n_workers"]))
    logger.info("Building data loader with {} workers.".format(cfg["n_workers"]))

    return data_loader
