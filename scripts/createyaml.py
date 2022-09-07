# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import yaml


def create_novel(baseyml, size, branch, outputdir):
    with open(baseyml, "r") as f:
        doc = yaml.full_load(f)

    doc["seed"] = 1
    arch = doc["model"]["feature_extractor"]["arch"]
    if arch[-2:] != str(size):
        raise Exception("Incorrect base yml {baseyml} for size {size}".format(baseyml=baseyml, size=size))

    if branch == "fc":
        doc["model"]["feature_extractor"]["n_freeze"] = 4
    else:
        doc["model"]["feature_extractor"]["n_freeze"] = 3

    doc["model"]["classifier"]["n_class"] = 40
    doc["data"]["loader"] = "NshotDataLoader"
    doc["data"]["train"] = "prepro/splits/imagenet/split80040/novel_train_40.csv"
    doc["data"]["val"] = "prepro/splits/imagenet/split80040/novel_val_40.csv"
    doc["data"]["n_shot"] = 0
    doc["data"]["lbl_offset"] = 0

    doc["training"]["epoch"] = 30
    doc["training"]["save_interval"] = 10
    doc["training"]["scheduler"]["step_size"] = 10
    doc["training"]["resume"]["model"] = "runs/imnet_base_{arch}/{arch}_800_norm4/ep-90_model.pkl".format(arch=arch)
    doc["exp"] = "FT_{arch}_800_norm4_{branch}_all_s1".format(arch=arch, branch=branch)

    with open("{}/imnet_novel_{}_{}.yml".format(outputdir, arch, branch), "w") as f:
        yaml.dump(doc, f)


def create_delta(baseyml, size, branch, outputdir):
    with open(baseyml, "r") as f:
        doc = yaml.full_load(f)

    doc["seed"] = 1
    arch = doc["model"]["feature_extractor"]["arch"]
    if arch[-2:] != str(size):
        raise Exception("Incorrect base yml {baseyml} for size {size}".format(baseyml=baseyml, size=size))

    doc["model"]["feature_extractor"]["arch"] = arch + "_2br"
    doc["model"]["feature_extractor"]["n_freeze"] = 5
    doc["model"]["classifier"]["arch"] = "deltacls"
    doc["model"]["classifier"]["n_bclass"] = 800
    doc["model"]["classifier"]["n_nclass"] = 40
    del doc["model"]["classifier"]["n_class"]
    doc["data"]["loader"] = "NshotDataLoader"
    doc["data"]["train"] = "prepro/splits/imagenet/split80040/train_40.csv"
    doc["data"]["val"] = "prepro/splits/imagenet/split80040/val_40-dev.csv"
    doc["data"]["n_shot"] = 10
    doc["training"]["epoch"] = 10
    doc["training"]["save_interval"] = 10
    doc["training"]["n_base_cls"] = 800
    doc["training"]["scheduler"]["step_size"] = 10
    doc["training"]["resume"]["model1"] = "runs/imnet_base_{arch}/{arch}_800_norm4/ep-90_model.pkl".format(arch=arch)
    doc["training"]["resume"][
        "model2"
    ] = "runs/imnet_novel_{arch}_{branch}/FT_{arch}_800_norm4_{branch}_all_s1/ep-30_model.pkl".format(
        arch=arch, branch=branch
    )
    doc["training"]["dp_scaling"] = "no_dp"
    del doc["training"]["resume"]["model"]
    doc["exp"] = (
        "split80040/{arch}_2br_800_{branch}_".format(arch=arch, branch=branch) + "40_a{:g}dp{:g}_10shot_s{:d}-dev"
    )

    with open("{}/imnet_delta_{}_{}.yml".format(outputdir, arch, branch), "w") as f:
        yaml.dump(doc, f)


def create_test(size, branch, outputdir):
    doc = dict()
    doc["seed"] = 1
    doc["model"] = dict()
    doc["model"] = dict()
    doc["model"]["feature_extractor"] = dict()
    doc["model"]["feature_extractor"]["arch"] = "resnet{size}_2br".format(size=size)
    doc["model"]["classifier"] = dict()
    doc["model"]["classifier"]["arch"] = "deltapoolcls"
    doc["model"]["classifier"]["feat_size"] = 512
    doc["model"]["classifier"]["n_bclass"] = 800
    doc["model"]["classifier"]["n_nclass"] = 40
    doc["model"]["classifier"]["norm"] = 4

    doc["data"] = dict()
    doc["data"]["loader"] = "BaseDataLoader"
    doc["data"]["root_dir"] = "prepro/data/imagenet"
    doc["data"]["val"] = "prepro/splits/imagenet/split80040/val_40-test.csv"
    doc["data"]["n_workers"] = 12

    doc["config"] = dict()
    doc["config"]["n_base_cls"] = 800
    doc["config"]["n_sel_cls"] = 800
    doc["config"]["norm"] = 4

    doc["checkpoint"] = dict()
    doc["checkpoint"][
        "model1"
    ] = "runs/imnet_delta_resnet{size}_{branch}/split80040/resnet{size}_2br_800_{branch}_40_a0.4dp0.2_10shot_s3-dev/ep-10_model.pkl".format(
        branch=branch, size=size
    )
    with open("{}/test_imnet_delta_resnet{}_{}.yml".format(outputdir, size, branch), "w") as f:
        yaml.dump(doc, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--baseyml",
        type=str,
        help="base yaml file",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=10,
        help="network size (10 or 18)",
    )
    parser.add_argument(
        "--branch",
        type=str,
        help="branch layer (layer4 or fc)",
    )
    parser.add_argument(
        "--outputdir",
        type=str,
        help="output dir for config files",
    )

    args = parser.parse_args()

    if args.branch != "fc" and args.branch != "layer4":
        raise Exception("Incorrect branch argument {}".format(args.branch))
    if args.size != 10 and args.size != 18:
        raise Exception("Incorrect network size argument {}".format(args.size))
    if not os.path.exists(args.baseyml):
        raise Exception("Base yaml file {} does not exist".format(args.baseyml))

    create_delta(args.baseyml, args.size, args.branch, args.outputdir)
    create_novel(args.baseyml, args.size, args.branch, args.outputdir)
    create_test(args.size, args.branch, args.outputdir)
