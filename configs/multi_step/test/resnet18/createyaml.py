# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import yaml
import string

with open("deltacls_branches_tmpl.yaml", "rt") as fp:
    cfg = yaml.load(fp, Loader=yaml.SafeLoader)

cfg_data_val_tmpl = string.Template(cfg["data"]["val"])
for x in range(1, 11):
    cfg["model"]["feature_extractor"]["n_branch"] = x + 1
    cfg["model"]["classifier"]["n_classes"] = [500] + [50] * x
    cfg["config"]["n_base_cls"] = [500] + [50] * (x - 1)
    cfg["data"]["val"] = cfg_data_val_tmpl.substitute(n_novel=50 * x)
    with open("deltacls_branches_split%d.yaml" % x, "wt") as fp:
        fp.write(yaml.dump(cfg, sort_keys=False))
