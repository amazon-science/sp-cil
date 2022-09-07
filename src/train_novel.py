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
    global norm

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
    params = [x for x in model_fe.parameters() if x.requires_grad] + list(model_cls.parameters())
    print("Trainable parameters: {}".format(sum([x.flatten().shape[0] for x in params])))
    logger.info("Trainable parameters: {}".format(sum([x.flatten().shape[0] for x in params])))
    opt_main = opt_main_cls(params, **opt_main_params)
    logger.info("Using optimizer {}".format(opt_main))

    # setup scheduler
    scheduler = step_scheduler(opt_main, **cfg["training"]["scheduler"])

    # load checkpoint
    start_ep = 0
    if cfg["training"]["resume"].get("model", None):
        resume = cfg["training"]["resume"]
        if os.path.isfile(resume["model"]):
            print("Loading model from checkpoint '{}'".format(resume["model"]))
            logger.info("Loading model from checkpoint '{}'".format(resume["model"]))
            checkpoint = torch.load(resume["model"])
            model_fe.module.load_state_dict(cvt2normal_state(checkpoint["model_fe_state"]))
            if resume.get("load_cls", False):
                model_cls.module.load_state_dict(cvt2normal_state(checkpoint["model_cls_state"]))
                logger.info("Loading classifier")
            if resume["param_only"] is False:
                start_ep = checkpoint["epoch"]
                opt_main.load_state_dict(checkpoint["opt_main_state"])
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            logger.info("Loaded checkpoint '{}' (iter {})".format(resume["model"], checkpoint["epoch"]))
        else:
            print("No checkpoint found at '{}'".format(resume["model"]))
            logger.info("No checkpoint found at '{}'".format(resume["model"]))

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
    top1 = averageMeter()
    top5 = averageMeter()

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

        # forward
        imfeat = model_fe(image)
        if norm > 0:
            imfeat = F.normalize(imfeat, p=2, dim=1) * norm
        output = model_cls(imfeat)

        loss = torch.mean(criterion(output, target).squeeze())
        losses.update(loss.item(), image.size(0))

        # measure accuracy
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], image.size(0))
        top5.update(prec5[0], image.size(0))

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
                "Top1 {top1.avg:.3f} "
                "Top5 {top5.avg:.3f}".format(
                    epoch + 1,
                    cfg["training"]["epoch"],
                    step + 1,
                    n_step,
                    curr_lr_main,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
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
            "Top1 {top1.avg:.3f} "
            "Top5 {top5.avg:.3f}".format(
                epoch + 1,
                cfg["training"]["epoch"],
                curr_lr_main,
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1,
                top5=top5,
            )
        )

        print(print_str)
        logger.info(print_str)
        writer.add_scalar("train/lr", curr_lr_main, epoch + 1)
        writer.add_scalar("train/loss", losses.avg, epoch + 1)
        writer.add_scalar("train/top1", top1.avg, epoch + 1)
        writer.add_scalar("train/top5", top5.avg, epoch + 1)


def val(data_loader, model_fe, model_cls, epoch, criterion):

    # setup average meters
    losses = averageMeter()
    top1 = averageMeter()
    top5 = averageMeter()

    # setting evaluation mode
    model_fe.eval()
    model_cls.eval()

    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        target = value[1].cuda(non_blocking=True)

        # forward
        imfeat = model_fe(image)
        if norm > 0:
            imfeat = F.normalize(imfeat, p=2, dim=1) * norm
        output = model_cls(imfeat)

        loss = torch.mean(criterion(output, target).squeeze())
        losses.update(loss.item(), image.size(0))

        # measure accuracy
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], image.size(0))
        top5.update(prec5[0], image.size(0))

    print_str = "[Val] Prec@1 {top1.avg:.3f}\t Prec@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
    print(print_str)
    logger.info(print_str)

    writer.add_scalar("val/loss", losses.avg, epoch + 1)
    writer.add_scalar("val/top1", top1.avg, epoch + 1)
    writer.add_scalar("val/top5", top5.avg, epoch + 1)


if __name__ == "__main__":
    global cfg, args, writer, logger

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/imnet_novel.yml",
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
