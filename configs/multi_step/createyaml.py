# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import yaml


def create_novel(baseyml, size, branch, outputdir):
    with open(baseyml, "r") as f:
        doc = yaml.full_load(f)

    for step in range(50, 550, 50):
        doc["seed"] = 1
        arch = doc["model"]["feature_extractor"]["arch"]
        if arch[-2:] != str(size):
            raise Exception("Incorrect base yml {baseyml} for size {size}".format(baseyml=baseyml, size=size))

        if branch == "fc":
            doc["model"]["feature_extractor"]["n_freeze"] = 4
        else:
            doc["model"]["feature_extractor"]["n_freeze"] = 3

        doc["model"]["classifier"]["n_class"] = 50
        doc["data"]["loader"] = "NshotDataLoader"
        doc["data"]["train"] = "prepro/splits/imagenet/split500/novel_train_{step}.csv".format(step=step)
        doc["data"]["val"] = "prepro/splits/imagenet/split500/novel_val_{step}.csv".format(step=step)
        doc["data"]["n_shot"] = 0
        doc["data"]["lbl_offset"] = 0

        doc["training"]["epoch"] = 30
        doc["training"]["save_interval"] = 10
        doc["training"]["scheduler"]["step_size"] = 10
        doc["training"]["resume"]["model"] = "runs/imnet_base_{arch}_500/{arch}_500_norm4/ep-90_model.pkl".format(
            arch=arch
        )
        doc["exp"] = "FT_{arch}_500_norm4_{branch}_all_s{step}".format(arch=arch, branch=branch, step=step)

        with open("{}/imnet_novel_{}_{}_{}.yml".format(outputdir, arch, branch, step), "w") as f:
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

    create_novel(args.baseyml, args.size, args.branch, args.outputdir)
