# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import argparse
import os
import yaml
import shutil
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from loader import get_dataloader
from models import get_model
from optimizers import get_optimizer, step_scheduler
from metrics import averageMeter, accuracy
from utils import get_logger, cvt2normal_state

from tensorboardX import SummaryWriter


def main():
    global norm, n_base_cls, n_novel_cls, n_overlap_cls, n_sel_cls

    if not torch.cuda.is_available():
        raise SystemExit("GPU is needed.")

    # setup random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup data loader
    splits = ["train", "val"]
    data_loader = get_dataloader(cfg["data"], splits, cfg["training"]["batch_size"])

    # config
    n_novel_cls = cfg["model"]["classifier"].get("n_nclass", 40)
    n_overlap_cls = cfg["model"]["classifier"].get("n_overlap", 0)
    n_base_cls = cfg["training"].get("n_base_cls", 800)
    n_sel_cls = cfg["training"].get("n_sel_cls", n_base_cls)

    # setup model (feature extractor + classifier)
    n_gpu = torch.cuda.device_count()
    model_fe = get_model(cfg["model"]["feature_extractor"], verbose=True).cuda()
    model_fe = nn.DataParallel(model_fe, device_ids=range(n_gpu))

    model_cls = get_model(cfg["model"]["classifier"], verbose=True).cuda()
    model_cls = nn.DataParallel(model_cls, device_ids=range(n_gpu))
    print("{} gpu(s) available.".format(n_gpu))

    # loss function
    criterion = nn.CrossEntropyLoss(reduction="none")
    norm = cfg["training"].get("norm", -1)

    # setup optimizer
    opt_main_cls, opt_main_params = get_optimizer(cfg["training"]["optimizer_main"])
    cnn_params = list(model_fe.parameters()) + list(model_cls.parameters())
    opt_main = opt_main_cls(cnn_params, **opt_main_params)
    logger.info("Using optimizer {}".format(opt_main))

    # setup scheduler
    scheduler = step_scheduler(opt_main, **cfg["training"]["scheduler"])

    # load checkpoint
    start_ep = 0
    if cfg["training"]["resume"].get("model1", False) and cfg["training"]["resume"].get("model2", False):
        model1 = cfg["training"]["resume"]["model1"]
        model2 = cfg["training"]["resume"]["model2"]
        if not os.path.isfile(model1):
            print("No checkpoint found at '{}'".format(model1))
            logger.info("No checkpoint found at '{}'".format(model1))
        elif not os.path.isfile(model2):
            print("No checkpoint found at '{}'".format(model2))
            logger.info("No checkpoint found at '{}'".format(model2))
        else:
            print("Loading model from checkpoint '{}' and '{}'".format(model1, model2))
            logger.info("Loading model from checkpoint '{}' and '{}'".format(model1, model2))
            checkpoint1 = torch.load(model1)
            checkpoint2 = torch.load(model2)
            model_fe.module.load_state_dict2(
                cvt2normal_state(checkpoint1["model_fe_state"]),
                cvt2normal_state(checkpoint2["model_fe_state"]),
            )
            model_cls.module.load_state_dict2(
                cvt2normal_state(checkpoint1["model_cls_state"]),
                cvt2normal_state(checkpoint2["model_cls_state"]),
            )
            logger.info("Loading classifier")

    print("Start training from epoch {}".format(start_ep))
    logger.info("Start training from epoch {}".format(start_ep))

    for ep in range(start_ep, cfg["training"]["epoch"]):

        train(data_loader["train"], model_fe, model_cls, opt_main, ep, criterion)

        if (ep + 1) % cfg["training"]["val_interval"] == 0:
            with torch.no_grad():
                val(data_loader["val"], model_fe, model_cls, ep, criterion)

        if (ep + 1) % cfg["training"]["save_interval"] == 0:
            state = {
                "epoch": ep + 1,
                "model_fe_state": model_fe.state_dict(),
                "model_cls_state": model_cls.state_dict(),
                "opt_main_state": opt_main.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            ckpt_path = os.path.join(writer.file_writer.get_logdir(), "ep-{ep}_model.pkl")
            save_path = ckpt_path.format(ep=ep + 1)
            last_path = ckpt_path.format(ep=ep + 1 - cfg["training"]["save_interval"])
            torch.save(state, save_path)
            if os.path.isfile(last_path):
                os.remove(last_path)
            print_str = "[Checkpoint]: {} saved".format(save_path)
            print(print_str)
            logger.info(print_str)

        scheduler.step()


def train(data_loader, model_fe, model_cls, opt_main, epoch, criterion):

    # setup average meters
    batch_time = averageMeter()
    data_time = averageMeter()
    losses = averageMeter()
    acc = averageMeter()
    bacc = averageMeter()
    nacc = averageMeter()
    oacc = averageMeter()

    # setting training mode
    model_fe.train()
    model_cls.train()

    n_step = int(len(data_loader.dataset) // float(data_loader.batch_size))
    end = time.time()
    for (step, value) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        image = value[0].cuda()
        target = value[1].cuda(non_blocking=True)
        isnovel = target >= n_base_cls
        isoverlap = target < n_overlap_cls
        isbase = (~isnovel) * (~isoverlap)
        target[isnovel] -= n_base_cls - n_sel_cls  # for partial selection (e.g. 40 / 800)

        # split label
        if len(value) > 3:
            sp_lbl = value[3].cuda()
        else:
            sp_lbl = isnovel.long()

        # forward
        _, feat1, feat2 = model_fe(image, feat=True)
        if norm > 0:
            feat1 = F.normalize(feat1, p=2, dim=1) * norm
            feat2 = F.normalize(feat2, p=2, dim=1) * norm

        # hn dropout
        dp_scaling = cfg["training"].get("dp_scaling", True)
        if dp_scaling is True:
            feat2[(sp_lbl == 0), :] = F.dropout(feat2[(sp_lbl == 0), :], p=dp) * (1 - dp)
        elif dp_scaling == "no_dp":
            feat2[(sp_lbl == 0), :] = feat2[(sp_lbl == 0), :] * (1 - dp)
        else:
            assert not dp_scaling
            feat2[(sp_lbl == 0), :] = F.dropout(feat2[(sp_lbl == 0), :], p=dp)

        output, route_out = model_cls(feat1, feat2, route=True)

        # compute loss
        r_ce = criterion(route_out, sp_lbl).squeeze()
        ce_b = r_ce[(sp_lbl == 0)].mean()
        ce_n = r_ce[(sp_lbl > 0)].mean()
        rloss = (ce_b + ce_n) / 2
        closs = torch.mean(criterion(output, target).squeeze())

        loss = (1 - alpha) * closs + alpha * rloss
        losses.update(loss.item(), image.size(0))

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

        # back propagation
        opt_main.zero_grad()
        loss.backward()
        opt_main.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % 10 == 0:
            curr_lr_main = opt_main.param_groups[0]["lr"]
            print_str = (
                "Epoch [{0}/{1}] "
                "Step: [{2}/{3}] "
                "LR: [{4}] "
                "Time {batch_time.avg:.3f} "
                "Data {data_time.avg:.3f} "
                "Loss {loss.avg:.4f} "
                "Acc {acc.avg:.3f} "
                "BaseAcc {bacc.avg:.3f} "
                "NovelAcc {nacc.avg:.3f} "
                "OverlapAcc {oacc.avg:.3f}".format(
                    epoch + 1,
                    cfg["training"]["epoch"],
                    step + 1,
                    n_step,
                    curr_lr_main,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=acc,
                    bacc=bacc,
                    nacc=nacc,
                    oacc=oacc,
                )
            )

            print(print_str)
            logger.info(print_str)

    if (epoch + 1) % cfg["training"]["print_interval"] == 0:
        curr_lr_main = opt_main.param_groups[0]["lr"]
        print_str = (
            "Epoch: [{0}/{1}] "
            "LR: [{2}] "
            "Time {batch_time.avg:.3f} "
            "Data {data_time.avg:.3f} "
            "Loss {loss.avg:.4f} "
            "Acc {acc.avg:.3f} "
            "BaseAcc {bacc.avg:.3f} "
            "NovelAcc {nacc.avg:.3f} "
            "OverlapAcc {oacc.avg:.3f}".format(
                epoch + 1,
                cfg["training"]["epoch"],
                curr_lr_main,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=acc,
                bacc=bacc,
                nacc=nacc,
                oacc=oacc,
            )
        )

        print(print_str)
        logger.info(print_str)
        writer.add_scalar("train/lr", curr_lr_main, epoch + 1)
        writer.add_scalar("train/loss", losses.avg, epoch + 1)
        writer.add_scalar("train/acc", acc.avg, epoch + 1)
        writer.add_scalar("train/bacc", bacc.avg, epoch + 1)
        writer.add_scalar("train/nacc", nacc.avg, epoch + 1)
        writer.add_scalar("train/oacc", oacc.avg, epoch + 1)


def val(data_loader, model_fe, model_cls, epoch, criterion):

    # setup average meters
    losses = averageMeter()
    racc = averageMeter()
    acc = averageMeter()
    bacc = averageMeter()
    nacc = averageMeter()
    oacc = averageMeter()
    base2novel = averageMeter()
    novel2base = averageMeter()

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
        target[isnovel] -= n_base_cls - n_sel_cls  # for partial selection (e.g. 40 / 800)

        # forward
        _, feat1, feat2 = model_fe(image, feat=True)
        if norm > 0:
            feat1 = F.normalize(feat1, p=2, dim=1) * norm
            feat2 = F.normalize(feat2, p=2, dim=1) * norm
        output = model_cls(feat1, feat2)

        loss = torch.mean(criterion(output, target).squeeze())
        losses.update(loss.item(), image.size(0))

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
        b2n = (pred[~isnovel] >= n_sel_cls).float().mean() if (n_base + n_overlap) > 0 else one
        n2b = (pred[isnovel] < n_sel_cls).float().mean() if n_novel > 0 else one
        if (n_base + n_overlap) > 0:
            base2novel.update(b2n.item(), n_base + n_overlap)
        if n_novel > 0:
            novel2base.update(n2b.item(), n_novel)
        r_acc = ((1 - b2n) * (n_base + n_overlap) + (1 - n2b) * n_novel) / image.size(0)
        racc.update(r_acc.item(), image.size(0))

    print_str = (
        "[Val] Acc {acc.avg:.4f} "
        "Racc {racc.avg: .3f} "
        "Bacc {bacc.avg: .4f} "
        "Nacc {nacc.avg: .4f} "
        "Oacc {oacc.avg: .4f} "
        "Base2novel {b2n.avg:.3f} "
        "Novel2base {n2b.avg:.3f}".format(
            acc=acc,
            racc=racc,
            bacc=bacc,
            nacc=nacc,
            oacc=oacc,
            b2n=base2novel,
            n2b=novel2base,
        )
    )
    print(print_str)
    logger.info(print_str)

    writer.add_scalar("val/loss", losses.avg, epoch + 1)
    writer.add_scalar("val/acc", acc.avg, epoch + 1)
    writer.add_scalar("val/racc", racc.avg, epoch + 1)
    writer.add_scalar("val/bacc", bacc.avg, epoch + 1)
    writer.add_scalar("val/nacc", nacc.avg, epoch + 1)
    writer.add_scalar("val/oacc", oacc.avg, epoch + 1)
    writer.add_scalar("val/base2novel", base2novel.avg, epoch + 1)
    writer.add_scalar("val/novel2base", novel2base.avg, epoch + 1)


if __name__ == "__main__":
    global cfg, args, writer, logger
    global alpha, dp, seed

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/imnet_delta.yml",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="weight for routing loss",
    )
    parser.add_argument(
        "--dp",
        type=float,
        default=None,
        help="hn dropout rate",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    seed = args.seed if args.seed else cfg.get("seed", 1)
    alpha = args.alpha if args.alpha else cfg.get("alpha", 0)
    dp = args.dp if args.dp else cfg.get("dp", 0)

    exp = cfg["exp"].format(alpha, dp, seed)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], exp)
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Start logging")

    print(args)
    logger.info(args)

    main()
