# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections import defaultdict
import numpy as np

from utils import *

### original splits of ImageNet
data_dir = "data/imagenet"
train = read_imagenet_data(data_dir, "imagenet_train.csv")
val = read_imagenet_data(data_dir, "imagenet_val.csv")
cls2id = get_imagenet_classdict(data_dir, "imagenet_classes.csv")

### base / novel class splits
seed = 1357
np.random.seed(seed)
perm = np.random.permutation(1000)
base_cls = perm[:800]
novel_cls = perm[800:]
n_novel = [40, 80]

# ImageNet id to base/novel class id
id2bcid = {x: i for i, x in enumerate(base_cls)}
id2ncid = {x: i for i, x in enumerate(novel_cls)}
id2allcid = {x: i for i, x in enumerate(perm)}

### filter images for each split and relabel them with base/novel class Id
tgt_dir = "splits/imagenet/split80040"
if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

# train
csv_file = os.path.join(tgt_dir, "base_train.csv")
gen_split(csv_file, train, base_cls, cls2id, id2bcid, prefix="train")

start = 0
for n in n_novel:
    # only novel
    csv_file = os.path.join(tgt_dir, "novel_train_{}.csv".format(n))
    gen_split(csv_file, train, novel_cls[start:n], cls2id, id2ncid, prefix="train")

    # base + novel
    csv_file = os.path.join(tgt_dir, "train_{}.csv".format(n))
    cls_list = np.unique(base_cls.tolist() + novel_cls[:n].tolist())
    gen_split(csv_file, train, cls_list, cls2id, id2allcid, prefix="train")

    start = n

# val
csv_file = os.path.join(tgt_dir, "base_val.csv")
gen_split(csv_file, val, base_cls, cls2id, id2bcid, prefix="val")

start = 0
for n in n_novel:
    # only novel
    csv_file = os.path.join(tgt_dir, "novel_val_{}.csv".format(n))
    gen_split(csv_file, val, novel_cls[start:n], cls2id, id2ncid, prefix="val")

    # base + novel
    csv_file = os.path.join(tgt_dir, "val_{}.csv".format(n))
    cls_list = np.unique(base_cls.tolist() + novel_cls[:n].tolist())
    gen_split(csv_file, val, cls_list, cls2id, id2allcid, prefix="val")

    start = 0

### split validation set into dev and test
np.random.seed(seed)

# accumulate image id for each class
cls2imgid_val = defaultdict(list)
for idx, cls in enumerate(val["LabelName"]):
    cls2imgid_val[cls2id[cls]].append(idx)

# split image id into two disjoint sets
for cls in range(1000):
    idx_list = cls2imgid_val.get(cls)
    n_sample = len(idx_list)
    np.random.shuffle(idx_list)
    cls2imgid_val[cls] = [idx_list[: n_sample // 2], idx_list[n_sample // 2 :]]

### 800 base classes
csv_file = os.path.join(tgt_dir, "base_val-dev.csv")
idx_list1 = np.concatenate([cls2imgid_val[x][0] for x in base_cls]).tolist()
gen_split(csv_file, val, base_cls, cls2id, id2bcid, prefix="val", allowed_idx=idx_list1)

csv_file = os.path.join(tgt_dir, "base_val-test.csv")
idx_list2 = np.concatenate([cls2imgid_val[x][1] for x in base_cls]).tolist()
gen_split(csv_file, val, base_cls, cls2id, id2bcid, prefix="val", allowed_idx=idx_list2)

### 40 novel classes
# only novel
csv_file = os.path.join(tgt_dir, "novel_val_40-dev.csv")
cls_list = novel_cls[:40].tolist()
id2ncid = {x: i for i, x in enumerate(cls_list)}
idx_list1 = np.concatenate([cls2imgid_val[x][0] for x in cls_list]).tolist()
gen_split(csv_file, val, cls_list, cls2id, id2ncid, prefix="val", allowed_idx=idx_list1)

csv_file = os.path.join(tgt_dir, "novel_val_40-test.csv")
idx_list2 = np.concatenate([cls2imgid_val[x][1] for x in cls_list]).tolist()
gen_split(csv_file, val, cls_list, cls2id, id2ncid, prefix="val", allowed_idx=idx_list2)

# all base + novel
csv_file = os.path.join(tgt_dir, "val_40-dev.csv")
cls_list = base_cls.tolist() + novel_cls[:40].tolist()
id2cid = {x: i for i, x in enumerate(cls_list)}
idx_list1 = np.concatenate([cls2imgid_val[x][0] for x in cls_list]).tolist()
gen_split(csv_file, val, cls_list, cls2id, id2cid, prefix="val", allowed_idx=idx_list1)

csv_file = os.path.join(tgt_dir, "val_40-test.csv")
idx_list2 = np.concatenate([cls2imgid_val[x][1] for x in cls_list]).tolist()
gen_split(csv_file, val, cls_list, cls2id, id2cid, prefix="val", allowed_idx=idx_list2)

### 200 novel classes
# only novel
csv_file = os.path.join(tgt_dir, "novel_val_200-dev.csv")
cls_list = novel_cls[:200].tolist()
id2ncid = {x: i for i, x in enumerate(cls_list)}
idx_list1 = np.concatenate([cls2imgid_val[x][0] for x in cls_list]).tolist()
gen_split(csv_file, val, cls_list, cls2id, id2ncid, prefix="val", allowed_idx=idx_list1)

csv_file = os.path.join(tgt_dir, "novel_val_200-test.csv")
idx_list2 = np.concatenate([cls2imgid_val[x][1] for x in cls_list]).tolist()
gen_split(csv_file, val, cls_list, cls2id, id2ncid, prefix="val", allowed_idx=idx_list2)

# all base + novel
csv_file = os.path.join(tgt_dir, "val_200-dev.csv")
cls_list = base_cls.tolist() + novel_cls[:200].tolist()
id2cid = {x: i for i, x in enumerate(cls_list)}
idx_list1 = np.concatenate([cls2imgid_val[x][0] for x in cls_list]).tolist()
gen_split(csv_file, val, cls_list, cls2id, id2cid, prefix="val", allowed_idx=idx_list1)

csv_file = os.path.join(tgt_dir, "val_200-test.csv")
idx_list2 = np.concatenate([cls2imgid_val[x][1] for x in cls_list]).tolist()
gen_split(csv_file, val, cls_list, cls2id, id2cid, prefix="val", allowed_idx=idx_list2)

### balanced number of base and novel classes
for n_cls in [40, 200]:
    cls_list = base_cls[:n_cls].tolist() + novel_cls[:n_cls].tolist()
    idx_list1 = np.concatenate([cls2imgid_val[x][0] for x in cls_list]).tolist()
    csv_file = os.path.join(tgt_dir, "base{}_novel{}_val-dev.csv".format(n_cls, n_cls))
    gen_split(csv_file, val, cls_list, cls2id, id2allcid, prefix="val", allowed_idx=idx_list1)

    idx_list2 = np.concatenate([cls2imgid_val[x][1] for x in cls_list]).tolist()
    csv_file = os.path.join(tgt_dir, "base{}_novel{}_val-test.csv".format(n_cls, n_cls))
    gen_split(csv_file, val, cls_list, cls2id, id2allcid, prefix="val", allowed_idx=idx_list2)

### subset of base classes
n_base = [40, 100, 200, 400]
for n_cls in n_base:
    cls_list = base_cls[:n_cls].tolist()
    idx_list1 = np.concatenate([cls2imgid_val[x][0] for x in cls_list]).tolist()
    csv_file = os.path.join(tgt_dir, "base{}_val-dev.csv".format(n_cls))
    gen_split(csv_file, val, cls_list, cls2id, id2bcid, prefix="val", allowed_idx=idx_list1)

    idx_list2 = np.concatenate([cls2imgid_val[x][1] for x in cls_list]).tolist()
    csv_file = os.path.join(tgt_dir, "base{}_val-test.csv".format(n_cls))
    gen_split(csv_file, val, cls_list, cls2id, id2bcid, prefix="val", allowed_idx=idx_list2)
