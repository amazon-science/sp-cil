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
    global norm, n_base_cls, n_overlap_cls, n_sel_cls

    if not torch.cuda.is_available():
        raise SystemExit("GPU is needed.")

    # setup mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup data loader
    splits = ["val"]
    data_loader = get_dataloader(cfg["data"], splits, args.b)

    # config
    n_overlap_cls = cfg["model"]["classifier"].get("n_overlap", 0)
    n_base_cls = cfg["config"].get("n_base_cls", 800)
    n_sel_cls = cfg["config"].get("n_sel_cls", n_base_cls)
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

    ckpt2 = cfg["checkpoint"].get("model2", None)
    if ckpt2:
        if not os.path.isfile(ckpt2):
            raise BaseException("No checkpoint found at '{}'".format(ckpt2))

    if ckpt2 is None:
        checkpoint1 = torch.load(ckpt1)
        model_fe.module.load_state_dict(cvt2normal_state(checkpoint1["model_fe_state"]))
        model_cls.module.load_state_dict(cvt2normal_state(checkpoint1["model_cls_state"]))
        print("Loading model from checkpoint '{}'".format(ckpt1))
        if logger:
            logger.info("Loading model from checkpoint '{}'".format(ckpt1))
    else:
        checkpoint1 = torch.load(ckpt1)
        checkpoint2 = torch.load(ckpt2)
        model_fe.module.load_state_dict2(
            cvt2normal_state(checkpoint1["model_fe_state"]),
            cvt2normal_state(checkpoint2["model_fe_state"]),
        )
        model_cls.module.load_state_dict2(
            cvt2normal_state(checkpoint1["model_cls_state"]),
            cvt2normal_state(checkpoint2["model_cls_state"]),
        )
        print("Loading model from checkpoint '{}' and '{}'".format(ckpt1, ckpt2))
        if logger:
            logger.info("Loading model from checkpoint '{}' and '{}'".format(ckpt1, ckpt2))

    with torch.no_grad():
        val(data_loader["val"], model_fe, model_cls)


def val(data_loader, model_fe, model_cls):

    # setup average meters
    racc = averageMeter()
    acc = averageMeter()
    bacc = averageMeter()
    nacc = averageMeter()
    oacc = averageMeter()
    base2novel = averageMeter()
    novel2base = averageMeter()
    base_blogit = averageMeter()
    base_nlogit = averageMeter()
    novel_blogit = averageMeter()
    novel_nlogit = averageMeter()

    # setting evaluation mode
    model_fe.eval()
    model_cls.eval()

    one = torch.tensor([1]).cuda()
    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        target = value[1].cuda(non_blocking=True)
        isnovel = target >= n_base_cls
        isoverlap = target < n_overlap_cls
        isbase = (~isnovel) * (~isoverlap)
        target[isnovel] -= n_base_cls - n_sel_cls

        # forward
        _, feat1, feat2 = model_fe(image, feat=True)
        if norm > 0:
            feat1 = F.normalize(feat1, p=2, dim=1) * norm
            feat2 = F.normalize(feat2, p=2, dim=1) * norm
        output = model_cls(feat1, feat2)

        # measure accuracy
        conf, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        iscorrect = torch.eq(pred, target)
        all_acc = iscorrect.float().mean()
        acc.update(all_acc.item(), image.size(0))

        # measure base and novel accuracy
        n_base = isbase.long().sum()
        n_novel = isnovel.long().sum()
        n_overlap = isoverlap.long().sum()
        assert (n_base + n_novel + n_overlap) == image.size(0)
        b_acc = iscorrect[isbase].float().mean()
        n_acc = iscorrect[isnovel].float().mean()
        o_acc = iscorrect[isoverlap].float().mean()
        if n_base > 0:
            bacc.update(b_acc.item(), n_base)
        if n_novel > 0:
            nacc.update(n_acc.item(), n_novel)
        if n_overlap > 0:
            oacc.update(o_acc.item(), n_overlap)

        # other analysis
        b2n = (pred[~isnovel] >= n_sel_cls).float().mean() if n_base > 0 else one
        n2b = (pred[isnovel] < n_sel_cls).float().mean() if n_novel > 0 else one
        if n_base > 0:
            base2novel.update(b2n.item(), n_base)
        if n_novel > 0:
            novel2base.update(n2b.item(), n_novel)
        r_acc = ((1 - b2n) * (n_base + n_overlap) + (1 - n2b) * n_novel) / image.size(0)
        racc.update(r_acc.item(), image.size(0))

        flag_has_nlogit = n_sel_cls < output.shape[1]
        blogit = output[:, :n_sel_cls].max(dim=1)[0] / (norm**2)
        if flag_has_nlogit:
            nlogit = output[:, n_sel_cls:].max(dim=1)[0] / (norm**2)
        if n_base > 0:
            base_blogit.update(blogit[~isnovel].mean().item(), n_base)
            if flag_has_nlogit:
                base_nlogit.update(nlogit[~isnovel].mean().item(), n_base)
        if n_novel > 0:
            novel_blogit.update(blogit[isnovel].mean().item(), n_novel)
            if flag_has_nlogit:
                novel_nlogit.update(nlogit[isnovel].mean().item(), n_novel)

    print_str = (
        "[Val] Acc {acc.avg:.4f} "
        "Racc {racc.avg: .3f} "
        "Bacc {bacc.avg: .4f} "
        "Nacc {nacc.avg: .4f} "
        "Oacc {oacc.avg: .4f} "
        "Base2novel {b2n.avg:.3f} "
        "Novel2base {n2b.avg:.3f} "
        "Blogit [B/N] [{bbl.avg:.3f}/{bnl.avg:.3f}] "
        "Nlogit [B/N] [{nbl.avg:.3f}/{nnl.avg:.3f}]".format(
            acc=acc,
            racc=racc,
            bacc=bacc,
            nacc=nacc,
            oacc=oacc,
            b2n=base2novel,
            n2b=novel2base,
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
        help="checkpoint of base classifier / whole score fusion net",
    )
    parser.add_argument(
        "--checkpoint2",
        type=str,
        default=None,
        help="checkpoint of novel classifier",
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
        cfg["checkpoint"]["model2"] = args.checkpoint2

    if args.pool and cfg["model"]["classifier"].get("pool", None):
        cfg["model"]["classifier"]["pool"] = args.pool

    print(cfg)
    if logger:
        logger.info(cfg)

    main()
