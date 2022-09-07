# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import pandas as pd
from collections import defaultdict

from metrics import averageMeter
import collections


def parse_result(exp, mode=None, n_seed=1):
    assert mode in [None, "overlap", "ms"]
    path = os.path.dirname(exp)
    seed_range = range(1, n_seed + 1)
    alpha_range = [0, 0.2, 0.4, 0.6, 0.8, 1]
    dp_range = [0, 0.2, 0.4, 0.6, 0.8, 1]

    result = defaultdict(list)
    for alpha in alpha_range:
        for dp in dp_range:
            Acc = averageMeter()
            Bacc = averageMeter()
            Nacc = averageMeter()
            Oacc = averageMeter()
            SPacc = collections.defaultdict(averageMeter)
            Avg_acc = averageMeter()
            Area = averageMeter()
            for seed in seed_range:
                # read results from the logfile
                logdir = exp.format(alpha, dp, seed)
                log = sorted([x for x in os.listdir(logdir) if "run" in x])[-1]
                log = os.path.join(logdir, log)
                with open(log, "r") as f:
                    res = f.readlines()[-2]
                res = [x for x in res.split(" ") if x]

                # get metrics
                acc = float(res[res.index("Acc") + 1])
                if mode == "ms":
                    spacc = []
                    idx_spacc = res.index("SPacc") + 1
                    while True:
                        spacc_item = res[idx_spacc]
                        spacc.append(float(spacc_item.rstrip("/")))
                        if not spacc_item.endswith("/"):
                            break
                        idx_spacc += 1
                    avg_acc = sum(spacc) / len(spacc)
                else:
                    bacc = float(res[res.index("Bacc") + 1])
                    nacc = float(res[res.index("Nacc") + 1])
                    oacc = float(res[res.index("Oacc") + 1])
                    if mode == "overlap":
                        avg_acc = (bacc + nacc + oacc) / 3
                    else:
                        avg_acc = (bacc + nacc) / 2
                area = acc * avg_acc

                # append results
                Acc.update(acc, 1)
                if mode == "ms":
                    for i, spacc_item in enumerate(spacc):
                        SPacc[i].update(spacc_item, 1)
                else:
                    Bacc.update(bacc, 1)
                    Nacc.update(nacc, 1)
                    Oacc.update(oacc, 1)
                Avg_acc.update(avg_acc, 1)
                Area.update(area, 1)

            # append results
            result["alpha"].append(alpha)
            result["dp"].append(dp)
            result["Acc (all)"].append(Acc.avg)
            if mode == "ms":
                for i, SPacc_item in SPacc.items():
                    result["Acc ({i})".format(i=i)].append(SPacc_item.avg)
            else:
                result["Acc (base)"].append(Bacc.avg)
                result["Acc (novel)"].append(Nacc.avg)
                result["Acc (overlap)"].append(Oacc.avg)
            result["Avg. Acc"].append(Avg_acc.avg)
            result["Area"].append(Area.avg)

    pd.set_option("precision", 4)
    if mode == "ms":
        df = pd.DataFrame(
            result,
            columns=["alpha", "dp", "Acc (all)"] + ["Acc ({i})".format(i=i) for i in SPacc] + ["Avg. Acc", "Area"],
        )
    elif mode == "overlap":
        df = pd.DataFrame(
            result,
            columns=[
                "alpha",
                "dp",
                "Acc (all)",
                "Acc (base)",
                "Acc (novel)",
                "Acc (overlap)",
                "Avg. Acc",
                "Area",
            ],
        )
    else:
        df = pd.DataFrame(
            result,
            columns=[
                "alpha",
                "dp",
                "Acc (all)",
                "Acc (base)",
                "Acc (novel)",
                "Avg. Acc",
                "Area",
            ],
        )

    # hyper-parameter selection
    best_all_idx = df.iloc[:, 2].idxmax()
    best_avg_idx = df.iloc[:, -2].idxmax()
    best_area_idx = df.iloc[:, -1].idxmax()
    sel_idx = [best_all_idx, best_area_idx, best_avg_idx]
    df_sel = df.iloc[sel_idx]
    print(df_sel)

    # save searching results
    csv_file = os.path.join(path, "sel_result.csv")
    df_sel.to_csv(csv_file, index=False, float_format="%.4f")

    csv_file = os.path.join(path, "all_result.csv")
    df.to_csv(csv_file, index=False, float_format="%.4f")

    sel_hyper = df_sel.iloc[:, [0, 1]].to_dict("list")
    return list(zip(sel_hyper["dp"], sel_hyper["alpha"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--exp",
        type=str,
        default="runs/imnet_delta/split2/resnet10_2br_800_layer4_40_a{:g}dp{:g}_10shot_s{:g}-dev",
        help="experiment",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["overlap", "ms"],
        help="is overlapping case",
    )
    parser.add_argument(
        "--n_seed",
        type=int,
        default=1,
        help="number of seeds for hyperparameter search",
    )
    args = parser.parse_args()

    sel_hyper = parse_result(args.exp, args.mode, args.n_seed)
