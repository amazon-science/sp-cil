# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import yaml
import logging
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from loader import get_dataloader
from models import get_model
from metrics import averageMeter
from utils import cvt2normal_state


def main():
    # global norm, n_base_cls, n_overlap_cls, n_sel_cls
    global norm, n_base_cls, n_all_cls

    if not torch.cuda.is_available():
        raise SystemExit("GPU is needed.")

    # setup mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup data loader
    splits = ["val"]
    data_loader = get_dataloader(cfg["data"], splits, args.b)

    # config
    n_novel_cls = cfg["config"].get("n_novel_cls", 40)
    n_base_cls = cfg["config"].get("n_base_cls", [800])
    assert cfg["config"].get("n_sel_cls", n_base_cls) == n_base_cls
    assert cfg["config"].get("n_overlap", 0) == 0
    assert cfg["model"]["classifier"].get("n_overlap", 0) == 0
    n_all_cls = n_base_cls + [n_novel_cls]
    norm = cfg["config"].get("norm", -1)

    # setup model (feature extractor + classifier)
    n_gpu = torch.cuda.device_count()
    model_fe = get_model(cfg["model"]["feature_extractor"], verbose=True).cuda()
    model_fe = nn.DataParallel(model_fe, device_ids=range(n_gpu))

    model_cls = get_model(cfg["model"]["classifier"], verbose=True).cuda()
    model_cls = nn.DataParallel(model_cls, device_ids=range(n_gpu))
    print("{} gpu(s) available.".format(n_gpu))

    # load checkpoints
    ckpt1 = cfg["checkpoint"].get("model1", None)
    if ckpt1 is None:
        raise BaseException("Checkpoint needs to be specified.")

    if not os.path.isfile(ckpt1):
        raise BaseException("No checkpoint found at '{}'".format(ckpt1))

    checkpoint1 = torch.load(ckpt1)
    model_fe.module.load_state_dict(cvt2normal_state(checkpoint1["model_fe_state"]))
    model_cls.module.load_state_dict(cvt2normal_state(checkpoint1["model_cls_state"]))
    print("Loading model from checkpoint '{}'".format(ckpt1))
    if logger:
        logger.info("Loading model from checkpoint '{}'".format(ckpt1))

    with torch.no_grad():
        val(data_loader["val"], model_fe, model_cls)


def val(data_loader, model_fe, model_cls):

    # setup average meters
    racc = averageMeter()
    acc = averageMeter()
    spacc = [averageMeter() for _ in n_all_cls]
    base2novel = averageMeter()
    novel2base = averageMeter()
    base_blogit = averageMeter()
    base_nlogit = averageMeter()
    novel_blogit = averageMeter()
    novel_nlogit = averageMeter()

    # setting evaluation mode
    model_fe.eval()
    model_cls.eval()
    split_lookup = sum([[idx] * n for idx, n in enumerate(n_all_cls)], [])
    split_lookup = torch.tensor(split_lookup).long().cuda()

    one = torch.tensor([1]).cuda()
    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        target = value[1].cuda(non_blocking=True)
        split_ids = split_lookup[target]

        # forward
        _, *feats = model_fe(image, feat=True)
        if norm > 0:
            feats = [F.normalize(feat, p=2, dim=1) * norm for feat in feats]
        output = model_cls(*feats)

        # measure accuracy
        conf, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        iscorrect = torch.eq(pred, target)
        all_acc = iscorrect.float().mean()
        acc.update(all_acc.item(), image.size(0))

        # measure base and novel accuracy
        n_split_samples = []
        for idx_split, n_split_cls in enumerate(n_all_cls):
            is_curr_split = split_ids == idx_split
            n_split_sample = is_curr_split.long().sum()
            split_acc = iscorrect[is_curr_split].float().mean()
            if n_split_sample:
                spacc[idx_split].update(split_acc.item(), n_split_sample)
            n_split_samples.append(n_split_sample)
        assert sum(n_split_samples) == image.size(0)

        # other analysis
        n_splits = len(n_all_cls)
        pred_split = split_lookup[pred]
        n_split_samples_prv = sum(n_split_samples[:-1])

        b2n = (pred_split[split_ids < n_splits - 1] == n_splits - 1).float().mean() if n_split_samples_prv > 0 else one
        n2b = (pred_split[split_ids == n_splits - 1] < n_splits - 1).float().mean() if n_split_samples[-1] > 0 else one
        if n_split_samples_prv > 0:
            base2novel.update(b2n.item(), n_split_samples_prv)
        if n_split_samples[-1] > 0:
            novel2base.update(n2b.item(), n_split_samples[-1])
        r_acc = (pred_split == split_ids).float().mean()
        racc.update(r_acc.item(), image.size(0))

        n_last_novel_cls = n_all_cls[-1]
        flag_has_nlogit = bool(n_last_novel_cls)
        blogit = output[:, :-n_last_novel_cls].max(dim=1)[0] / (norm**2)
        if flag_has_nlogit:
            nlogit = output[:, -n_last_novel_cls:].max(dim=1)[0] / (norm**2)
        if n_split_samples_prv > 0:
            base_blogit.update(blogit[split_ids < n_splits - 1].mean().item(), n_split_samples_prv)
            if flag_has_nlogit:
                base_nlogit.update(nlogit[split_ids < n_splits - 1].mean().item(), n_split_samples_prv)
        if n_split_samples[-1] > 0:
            novel_blogit.update(blogit[split_ids == n_splits - 1].mean().item(), n_split_samples[-1])
            if flag_has_nlogit:
                novel_nlogit.update(nlogit[split_ids == n_splits - 1].mean().item(), n_split_samples[-1])

    print_str = (
        "[Val] Acc {acc.avg:.4f} "
        "Racc {racc.avg: .3f} "
        "SPacc {spacc} "
        "Base2novel {b2n.avg:.3f} "
        "Novel2base {n2b.avg:.3f} "
        "Blogit [B/N] [{bbl.avg:.3f}/{bnl.avg:.3f}] "
        "Nlogit [B/N] [{nbl.avg:.3f}/{nnl.avg:.3f}]".format(
            acc=acc,
            racc=racc,
            b2n=base2novel,
            n2b=novel2base,
            spacc="/".join("{: .4f}".format(x.avg) for x in spacc),
            bbl=base_blogit,
            bnl=base_nlogit,
            nbl=novel_blogit,
            nnl=novel_nlogit,
        )
    )
    print(print_str)
    if logger:
        logger.info(print_str)


if __name__ == "__main__":
    global cfg, args, logger

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        type=str,
        default="test_configs/split2_800_40/resnet10/poolcls.yaml",
        help="config file for testing",
    )
    parser.add_argument(
        "--checkpoint1",
        type=str,
        default=None,
        help="checkpoint of the whole score fusion net",
    )
    parser.add_argument(
        "--pool",
        type=str,
        default=None,
        help="max / avg",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="path to the logfile (default: None)",
    )
    parser.add_argument(
        "--b",
        type=int,
        default=256,
        help="batch size",
    )

    args = parser.parse_args()
    print(args)

    if args.log:
        logger = logging.getLogger("mylogger")
        hdlr = logging.FileHandler(args.log)
        logger.addHandler(hdlr)
        logger.setLevel(logging.INFO)
    else:
        logger = None

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    if args.checkpoint1:
        cfg["checkpoint"]["model1"] = args.checkpoint1

    if args.pool and cfg["model"]["classifier"].get("pool", None):
        cfg["model"]["classifier"]["pool"] = args.pool

    print(cfg)
    if logger:
        logger.info(cfg)

    main()
