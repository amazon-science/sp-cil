# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import os
from PIL import Image

import torch.utils.data as data


class ImageFilelist(data.Dataset):
    def __init__(self, root_dir, flist, transform=None, target_transform=None, load_sp_lbl=False):

        self.root_dir = root_dir  # root dir of images

        if load_sp_lbl:
            self.data = pd.read_csv(flist).iloc[:, [0, 3, 4, 5]].to_dict("list")  # only select the cols of interest
        else:
            self.data = pd.read_csv(flist).iloc[:, [0, 3, 4]].to_dict("list")  # only select the cols of interest
        self.transform = transform
        self.target_transform = target_transform
        self.load_sp_lbl = load_sp_lbl

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.data["ImageID"][index])
        img = Image.open(img_path).convert("RGB")
        target = self.data["ClsID"][index]
        idx = self.data["Idx"][index]
        if self.load_sp_lbl:
            sp_lbl = self.data["SPLbl"][index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.load_sp_lbl:
            return img, target, idx, sp_lbl
        else:
            return img, target, idx

    def __len__(self):
        return len(self.data["ImageID"])
