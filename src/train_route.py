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
    global norm, n_base_cls, n_novel_cls, n_overlap_cls, n_sel_cls, loss_type

    if not torch.cuda.is_available():
        raise SystemExit("GPU is needed.")

    # setup random seed
    torch.manual_seed(cfg.get("seed", 1))
    torch.cuda.manual_seed(cfg.get("seed", 1))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup data loader
    splits = ["train", "val"]
    data_loader = get_dataloader(cfg["data"], splits, cfg["training"]["batch_size"])

    # config
    n_base_cls = cfg["model"]["base_classifier"].get("n_class", 800)
    n_novel_cls = cfg["model"]["novel_classifier"].get("n_class", 40)
    n_overlap_cls = cfg["training"].get("n_overlap_cls", 0)
    n_sel_cls = cfg["training"].get("n_sel_cls", n_base_cls)

    # setup model (feature extractor + classifier)
    n_gpu = torch.cuda.device_count()
    model_fe = get_model(cfg["model"]["feature_extractor"], verbose=True).cuda()
    model_fe = nn.DataParallel(model_fe, device_ids=range(n_gpu))

    model_rcls = get_model(cfg["model"]["route_classifier"], verbose=True).cuda()
    model_rcls = nn.DataParallel(model_rcls, device_ids=range(n_gpu))

    model_bcls = get_model(cfg["model"]["base_classifier"], verbose=True).cuda()
    model_bcls = nn.DataParallel(model_bcls, device_ids=range(n_gpu))

    model_ncls = get_model(cfg["model"]["novel_classifier"], verbose=True).cuda()
    model_ncls = nn.DataParallel(model_ncls, device_ids=range(n_gpu))
    print("{} gpu(s) available.".format(n_gpu))

    # loss function
    criterion = nn.CrossEntropyLoss(reduction="none")
    norm = cfg["training"].get("norm", -1)
    loss_type = cfg["training"].get("loss_type", "balanced")

    # setup optimizer
    opt_main_cls, opt_main_params = get_optimizer(cfg["training"]["optimizer_main"])
    cnn_params = list(model_fe.parameters()) + list(model_rcls.parameters())
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
            model_bcls.module.load_state_dict(cvt2normal_state(checkpoint1["model_cls_state"]))
            model_ncls.module.load_state_dict(cvt2normal_state(checkpoint2["model_cls_state"]))
            logger.info("Loading classifier")

    print("Start training from epoch {}".format(start_ep))
    logger.info("Start training from epoch {}".format(start_ep))

    for ep in range(start_ep, cfg["training"]["epoch"]):

        train(data_loader["train"], model_fe, model_rcls, opt_main, ep, criterion)

        if (ep + 1) % cfg["training"]["val_interval"] == 0:
            with torch.no_grad():
                val(
                    data_loader["val"],
                    model_fe,
                    model_rcls,
                    model_bcls,
                    model_ncls,
                    ep,
                    criterion,
                )

        if (ep + 1) % cfg["training"]["save_interval"] == 0:
            state = {
                "epoch": ep + 1,
                "model_fe_state": model_fe.state_dict(),
                "model_rcls_state": model_rcls.state_dict(),
                "model_bcls_state": model_bcls.state_dict(),
                "model_ncls_state": model_ncls.state_dict(),
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


def train(data_loader, model_fe, model_rcls, opt_main, epoch, criterion):

    # setup average meters
    batch_time = averageMeter()
    data_time = averageMeter()
    losses = averageMeter()
    racc = averageMeter()
    base2novel = averageMeter()
    novel2base = averageMeter()

    # setting training mode
    model_fe.train()
    model_rcls.train()

    one = torch.tensor([1]).cuda()
    n_step = int(len(data_loader.dataset) // float(data_loader.batch_size))
    end = time.time()
    for (step, value) in enumerate(data_loader):
        if step == n_step:
            break
        # measure data loading time
        data_time.update(time.time() - end)

        image = value[0].cuda()
        target = value[1].cuda(non_blocking=True)
        isnovel = target >= n_base_cls

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
        imfeat = torch.cat([feat1, feat2], dim=1)
        output = model_rcls(imfeat)

        # compute loss
        if loss_type == "reweight":
            n_class = float(n_base_cls + n_novel_cls)
            weight = torch.tensor([n_class / n_base_cls, n_class / n_novel_cls]).cuda()
            alpha = weight.gather(0, sp_lbl.data)
            loss = torch.mean(criterion(output, sp_lbl).squeeze() * alpha)

        elif loss_type == "balanced":
            r_ce = criterion(output, sp_lbl).squeeze()
            ce_b = r_ce[(sp_lbl == 0)].mean()
            ce_n = r_ce[(sp_lbl > 0)].mean()
            loss = (ce_b + ce_n) / 2

        else:
            loss = torch.mean(criterion(output, sp_lbl).squeeze())

        losses.update(loss.item(), image.size(0))

        # measure accuracy
        conf, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        iscorrect = torch.eq(pred, sp_lbl)
        r_acc = iscorrect.float().mean()
        racc.update(r_acc.item(), image.size(0))

        # other analysis
        n_novel = isnovel.long().sum()
        n_base = image.size(0) - n_novel
        b2n = ((~isnovel) * (~iscorrect)).float().sum() / n_base if n_base > 0 else one
        n2b = (isnovel * (~iscorrect)).float().sum() / n_novel if n_novel > 0 else one
        if n_base > 0:
            base2novel.update(b2n.item(), n_base)
        if n_novel > 0:
            novel2base.update(n2b.item(), n_novel)

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
                "Acc {acc.avg:.3f}".format(
                    epoch + 1,
                    cfg["training"]["epoch"],
                    step + 1,
                    n_step,
                    curr_lr_main,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    acc=racc,
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
            "Acc {acc.avg:.3f}".format(
                epoch + 1,
                cfg["training"]["epoch"],
                curr_lr_main,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=racc,
            )
        )

        print(print_str)
        logger.info(print_str)
        writer.add_scalar("train/lr", curr_lr_main, epoch + 1)
        writer.add_scalar("train/loss", losses.avg, epoch + 1)
        writer.add_scalar("train/racc", racc.avg, epoch + 1)
        writer.add_scalar("train/base2novel", base2novel.avg, epoch + 1)
        writer.add_scalar("train/novel2base", novel2base.avg, epoch + 1)


def val(data_loader, model_fe, model_rcls, model_bcls, model_ncls, epoch, criterion):

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
    model_rcls.eval()
    model_bcls.eval()
    model_ncls.eval()

    zero = torch.tensor([0]).cuda()
    one = torch.tensor([1]).cuda()
    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        target = value[1].cuda(non_blocking=True)
        isnovel = target >= n_base_cls
        isoverlap = target < n_overlap_cls
        isbase = (~isnovel) * (~isoverlap)
        target[isnovel] -= n_base_cls  # for partial selection (e.g. 40 / 800)

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
        imfeat = torch.cat([feat1, feat2], dim=1)
        output = model_rcls(imfeat)
        out1 = model_bcls(feat1)
        out2 = model_ncls(feat2)

        loss = torch.mean(criterion(output, sp_lbl).squeeze())
        losses.update(loss.item(), image.size(0))

        # measure routing accuracy
        conf, pred = torch.max(torch.softmax(output, dim=1), dim=1)
        iscorrect = torch.eq(pred, sp_lbl)
        r_acc = iscorrect.float().mean()
        racc.update(r_acc.item(), image.size(0))

        # other analysis
        n_novel = isnovel.long().sum()
        n_base = image.size(0) - n_novel
        b2n = ((~isnovel) * (~iscorrect)).float().sum() / n_base if n_base > 0 else one
        n2b = (isnovel * (~iscorrect)).float().sum() / n_novel if n_novel > 0 else one
        if n_base > 0:
            base2novel.update(b2n.item(), n_base)
        if n_novel > 0:
            novel2base.update(n2b.item(), n_novel)

        # measure classification accuracy
        n_base = isbase.long().sum()
        b_idx = isbase * (pred == 0)
        _, pred1 = torch.max(torch.softmax(out1[:, :n_sel_cls], dim=1), dim=1)
        b_iscorrect = torch.eq(pred1[b_idx], target[b_idx])
        b_acc = b_iscorrect.float().sum() / n_base if n_base > 0 else zero

        n_idx = isnovel * (pred == 1)
        _, pred2 = torch.max(torch.softmax(out2, dim=1), dim=1)
        n_iscorrect = torch.eq(pred2[n_idx], target[n_idx] + n_overlap_cls)
        n_acc = n_iscorrect.float().sum() / n_novel if n_novel > 0 else zero

        n_overlap = isoverlap.long().sum()
        o_idx1 = isoverlap * (pred == 0)
        o_iscorrect1 = torch.eq(pred1[o_idx1], target[o_idx1])
        o_idx2 = isoverlap * (pred == 1)
        o_iscorrect2 = torch.eq(pred2[o_idx2], target[o_idx2])
        o_acc = (o_iscorrect1.float().sum() + o_iscorrect2.float().sum()) / n_overlap if n_overlap > 0 else zero

        assert (n_base + n_novel + n_overlap) == image.size(0)
        all_acc = (b_acc * n_base + n_acc * n_novel + o_acc * n_overlap) / image.size(0)
        if n_base > 0:
            bacc.update(b_acc.item(), n_base)
        if n_novel > 0:
            nacc.update(n_acc.item(), n_novel)
        if n_overlap > 0:
            oacc.update(o_acc.item(), n_overlap)
        acc.update(all_acc.item(), image.size(0))

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

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/imnet_base.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], cfg["exp"])
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Start logging")

    print(args)
    logger.info(args)

    main()
