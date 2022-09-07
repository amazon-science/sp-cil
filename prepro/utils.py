# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import pandas as pd


def read_imagenet_data(data_dir, csv_file):
    """Read ImageNet data (image name and labels) from csv_file
    and return a dictonary with keys 'ImageID' and 'LabelName'.
    """
    df = pd.read_csv(os.path.join(data_dir, csv_file), sep=",", header=0)
    return df.iloc[:, [0, 2]].to_dict("list")


def get_imagenet_classdict(data_dir, csv_file):
    """Read ImageNet class mapping (id->cls_name) and return a dictionary."""
    df = pd.read_csv(os.path.join(data_dir, csv_file), sep="\t", header=None)
    return {x[1]: x[0] for x in df.values}


def gen_split(
    csv_file,
    data,
    cls_list,
    cls2id,
    id_dict,
    prefix="",
    allowed_idx=None,
    store_sp_lbl=False,
):
    """Generate data splits for base/novel classes and save to csv_file."""
    split = []
    if store_sp_lbl:

        if allowed_idx is None:
            allowed_idx = [list(range(len(data["ImageID"])))]

        # store split label (0: base / 1: novel)
        for sp_lbl, idx_list in enumerate(allowed_idx):
            for idx in idx_list:
                cls = data["LabelName"][idx]
                lbl = cls2id[cls]
                if lbl not in cls_list:
                    continue

                if prefix == "val":
                    img_name = os.path.join(prefix, cls, data["ImageID"][idx])
                else:
                    img_name = os.path.join(prefix, data["ImageID"][idx])

                new_lbl = id_dict[lbl]
                split.append([img_name, cls, lbl, idx, new_lbl, sp_lbl])

        df = pd.DataFrame(split, columns=["ImageID", "LabelName", "Label", "Idx", "ClsID", "SPLbl"])

    else:

        if allowed_idx is None:
            allowed_idx = list(range(len(data["ImageID"])))

        for idx in allowed_idx:
            cls = data["LabelName"][idx]
            lbl = cls2id[cls]
            if lbl not in cls_list:
                continue

            if prefix == "val":
                img_name = os.path.join(prefix, cls, data["ImageID"][idx])
            else:
                img_name = os.path.join(prefix, data["ImageID"][idx])

            new_lbl = id_dict[lbl]
            split.append([img_name, cls, lbl, idx, new_lbl])
        df = pd.DataFrame(split, columns=["ImageID", "LabelName", "Label", "Idx", "ClsID"])

    df.to_csv(csv_file, index=False)
    print("{} generated.".format(csv_file))
