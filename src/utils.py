# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import logging
import datetime
from collections import OrderedDict


def adjust_learning_rate(optimizer, decay_rate=0.9):
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * decay_rate


def assign_learning_rate(optimizer, lr=0.1):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def add_weight_decay(params, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in params:
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": l2_value},
    ]


def get_logger(logdir):
    """Function to build the logger."""
    logger = logging.getLogger("mylogger")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def cvt2normal_state(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
    module state_dict inplace, i.e. removing "module" in the string.
    """
    new_state_dict = OrderedDict()
    for name, param in state_dict.items():
        name = name.replace("module.", "")
        new_state_dict[name] = param
    return new_state_dict
