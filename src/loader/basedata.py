# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import torch.utils.data as data
import torchvision.transforms as transforms

from .img_flist import ImageFilelist

logger = logging.getLogger("mylogger")


def BaseDataLoader(cfg, splits, batch_size):

    data_loader = dict()

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

        dataset = ImageFilelist(cfg["root_dir"], cfg[split], transform, load_sp_lbl=load_sp_lbl)
        data_loader[split] = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
            pin_memory=True,
            num_workers=cfg["n_workers"],
        )
        print("{split}: {size}".format(split=split, size=len(dataset)))
        logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    print("Building data loader with {} workers.".format(cfg["n_workers"]))
    logger.info("Building data loader with {} workers.".format(cfg["n_workers"]))

    return data_loader
