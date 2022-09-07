# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
import argparse
import yaml
import pandas as pd
from collections import defaultdict
from termcolor import colored

from metrics import averageMeter
from parse_result import parse_result


def hyper_search(config, exp, n_seed=1):
    alpha_range = [0, 0.2, 0.4, 0.6, 0.8, 1]
    dp_range = [0, 0.2, 0.4, 0.6, 0.8, 1]

    for seed in range(1, n_seed + 1):
        for dp in dp_range:
            for alpha in alpha_range:
                if not os.path.isdir(exp.format(alpha, dp, seed)):
                    print(train.format(config, seed, dp, alpha))
                    os.system(train.format(config, seed, dp, alpha))


def main():
    print(colored("[ Start hyper-parameter search ]", "cyan"))
    # run hyper-parameter search with development set
    exp = os.path.join("runs", os.path.basename(args.train_config)[:-4], cfg["exp"])
    hyper_search(args.train_config, exp, args.n_seed)

    # parse log and get the results of the selected hyper-parameters
    mode = None
    if cfg["model"]["classifier"].get("n_classes", None):
        mode = "ms"
    elif cfg["model"]["classifier"].get("n_overlap", 0) > 0:
        mode = "overlap"
    sel_hyper = parse_result(exp, mode, args.n_seed)
    sel_all = False
    if sel_all:
        alpha_range = [0, 0.4, 1]
        dp_range = [0, 0.2, 0.4, 0.6, 0.8, 1]
        sel_hyper = [(dp, alpha) for alpha in alpha_range for dp in dp_range]

    for seed in range(args.n_seed + 1, 4):
        for (dp, alpha) in set(sel_hyper):
            if not os.path.isdir(exp.format(alpha, dp, seed)):
                print(train.format(args.train_config, seed, dp, alpha))
                os.system(train.format(args.train_config, seed, dp, alpha))

    # evaluate on testing set
    print(colored("[ Start testing ]", "cyan"))
    result = defaultdict(list)
    for (dp, alpha) in sel_hyper:
        Acc = averageMeter()
        Bacc = averageMeter()
        Nacc = averageMeter()
        Oacc = averageMeter()
        SPacc = defaultdict(averageMeter)
        Avg_acc = averageMeter()
        for seed in range(1, max(3, args.n_seed) + 1):
            ckpt = os.path.join(exp.format(alpha, dp, seed), "ep-%d_model.pkl" % args.ep)
            # run testing
            logfile = os.path.join(os.path.dirname(ckpt), "test_result.log")
            if not os.path.isfile(logfile):
                print(test.format(args.test_config, ckpt, logfile))
                os.system(test.format(args.test_config, ckpt, logfile))

            # read results from the logfile
            with open(logfile, "r") as f:
                res = f.readlines()[-1]
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

    pd.set_option("precision", 4)
    if mode == "ms":
        df = pd.DataFrame(
            result,
            columns=["alpha", "dp", "Acc (all)"] + ["Acc ({i})".format(i=i) for i in SPacc] + ["Avg. Acc"],
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
            ],
        )
    print(colored("[ Result ]", "cyan"))
    print(df)

    # save searching results
    path = os.path.dirname(exp)
    csv_file = os.path.join(path, "test_result.csv")
    df.to_csv(csv_file, index=False, float_format="%.4f")
    print("result saved to {}.".format(csv_file))


if __name__ == "__main__":
    global args, cfg, train, test
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--script",
        type=str,
        default="",
        help="which script, train_fusion.py (default or train_fusion_ms.py",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/imnet_delta.yml",
        help="config file for training",
    )
    parser.add_argument(
        "--test_config",
        type=str,
        default="test_configs/split2_800_40/resnet10/deltapoolcls.yaml",
        help="config file for testing",
    )
    parser.add_argument(
        "--n_seed",
        type=int,
        default=1,
        help="number of seeds for hyperparameter search",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=256,
        help="batch size for testing",
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=10,
        help="epoch of fusion training",
    )

    args = parser.parse_args()

    train = "python src/train_fusion%s.py --config {} --seed {} --dp {} --alpha {}" % (args.script)
    test = "python src/test_fusion%s.py --config {} --checkpoint1 {} --log {} --b %d" % (args.script, args.b)

    if not os.path.isfile(args.train_config):
        raise BaseException("train_config: '{}' not found".format(args.train_config))

    if not os.path.isfile(args.test_config):
        raise BaseException("test_config: '{}' not found".format(args.test_config))

    with open(args.train_config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    main()
