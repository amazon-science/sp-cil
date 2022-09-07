# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import torch.nn as nn
import logging

from .resnet import resnet10, resnet18, resnet50
from .resnet_2br import (
    resnet10_2br,
    resnet10_nbr,
    resnet18_2br,
    resnet18_nbr,
    resnet50_2br,
    resnet10_2brl3,
)
from .linearcls import linearcls
from .deltacls import deltacls, deltaclsn
from .deltapoolcls import deltapoolcls

logger = logging.getLogger("mylogger")


def get_model(model_dict, verbose=False):

    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if "resnet" in name:
        model = model(**param_dict)
        model.fc = nn.Identity()

    else:
        model = model(**param_dict)

    if verbose:
        logger.info(model)

    return model


def _get_model_instance(name):
    try:
        return {
            "resnet10": resnet10,
            "resnet10_2br": resnet10_2br,
            "resnet10_nbr": resnet10_nbr,
            "resnet10_2brl3": resnet10_2brl3,
            "resnet18": resnet18,
            "resnet18_2br": resnet18_2br,
            "resnet18_nbr": resnet18_nbr,
            "resnet50": resnet50,
            "resnet50_2br": resnet50_2br,
            "linearcls": linearcls,
            "deltacls": deltacls,
            "deltaclsn": deltaclsn,
            "deltapoolcls": deltapoolcls,
        }[name]
    except:
        raise BaseException("Model {} not available".format(name))
