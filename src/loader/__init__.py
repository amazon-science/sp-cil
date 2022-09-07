# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from loader.nshotdatanovel import NshotNovelDataLoader
from .basedata import BaseDataLoader
from .nshotdata import NshotDataLoader
from .baldata import BalancedDataLoader
from .nshotdatanovel import NshotNovelDataLoader


def get_dataloader(cfg, splits, batch_size, seed=None):
    loader = _get_loader_instance(cfg["loader"])
    if loader is NshotDataLoader or loader is NshotNovelDataLoader:
        data_loader = loader(cfg, splits, batch_size, seed=seed)
    else:
        data_loader = loader(cfg, splits, batch_size)
    return data_loader


def _get_loader_instance(name):
    try:
        return {
            "BaseDataLoader": BaseDataLoader,
            "NshotDataLoader": NshotDataLoader,
            "NshotNovelDataLoader": NshotNovelDataLoader,
            "BalancedDataLoader": BalancedDataLoader,
        }[name]
    except:
        raise ("Loader type {} not available".format(name))
