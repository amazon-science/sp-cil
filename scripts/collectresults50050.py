# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import glob
import numpy as np
import pandas as pd
import re

argv = sys.argv[1:]
assert len(argv) == 2, "Please call using two parameters, arch and layers. {}".format(argv)
supported_input = {
    ("resnet10", "layer4"),
    ("resnet18", "layer4"),
    ("resnet10", "fc"),
    ("resnet18", "fc"),
}
assert tuple(argv) in supported_input, 'arch+layers combination "{}" not recognized. supported:\n{}'.format(
    argv, list(supported_input)
)

network, branch = argv
n_sp = 10
shot = "_20shot" if network == "resnet18" else ""
result_files_tmpl = "runs/imnet_delta_{branch}_{network}{shot}_500_split{{n}}/split500/test_result.csv".format(
    network=network, branch=branch, shot=shot
)
base_result_file = sorted(
    glob.glob("runs/imnet_base_{network}_500/{network}_500_norm4/run_*.log".format(network=network))
)[-1]

results = [pd.read_csv(result_files_tmpl.format(n=n)) for n in range(1, n_sp + 1)]
for result in results:
    result["method"] = ["best-all", "best-bal", "best-avg"]

# deal with base performance: extract from training logs
with open(base_result_file, "rt") as f:
    base_result_line = f.readlines()[-2]
base_accu = re.search(r"Prec@1\s([\d\.]+)\t\sPrec@5\s[\d\.]+$", base_result_line).group(1)
base_result = results[0].drop("Acc (1)", axis=1)
base_result.loc[:, [x for x in base_result.columns if "Acc" in x]] = float(base_accu) / 100
results = [base_result] + results

results_agg = pd.concat([result[["method", "Acc (all)", "Avg. Acc"]] for result in results])

results_mean = results_agg.groupby("method", sort=False).mean()
print((results_mean * 100).applymap("{:.2f}".format).to_string())
