# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from fcos_core.utils.comm import get_world_size, is_pytorch_1_1_0_or_later
from fcos_core.utils.metric_logger import MetricLogger

import my.custom as custom


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    distributed
):
    # tensorboard summary
    tb_summary_writer = SummaryWriter(cfg.OUTPUT_DIR)

    logger = logging.getLogger("fcos_core.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        # in pytorch >= 1.1.0, scheduler.step() should be run after optimizer.step()
        if not pytorch_1_1_0_or_later:
            scheduler.step()

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict, cosines = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        #　mma loss
        if cfg.mma_fpn:
            for name, m in model.module.backbone.fpn.named_modules():
                if isinstance(m, (custom.Con2d_Class, custom.Con2d_Head, nn.Conv2d)):
                    mmaloss = get_angular_loss(m.weight)
                    if cfg.mma_fpn_weight != 0:
                        losses = losses + cfg.mma_fpn_weight * mmaloss
        if cfg.mma_head:
            for name, m in model.module.rpn.head.named_modules():
                if isinstance(m, (custom.Con2d_Class, custom.Con2d_Head, nn.Conv2d)):
                    mmaloss = get_angular_loss(m.weight)
                    if cfg.mma_head_weight != 0:
                        losses = losses + cfg.mma_head_weight * mmaloss
        if cfg.mma_cls:
            mmaloss = get_angular_loss(model.module.rpn.head.cls_logits.weight)
            if cfg.mma_cls_weight != 0:
                losses = losses + cfg.mma_cls_weight * mmaloss

        # dma loss
        if cfg.dma and cfg.dma_weight != 0:
            for cosine in cosines:
                losses = losses + cfg.dma_weight * dma_loss(cosine.permute(0, 2, 3, 1).contiguous().view(-1, 80))

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if pytorch_1_1_0_or_later:
            scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

            tb_summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], iteration)
            tb_summary_writer.add_scalar('loss', losses_reduced, iteration)
            for k, v in loss_dict_reduced.items():
                tb_summary_writer.add_scalar(k, v, iteration)

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        if cfg.TENSORBOARD.PARAS and iteration % cfg.TENSORBOARD.PERIOD == 0:
            MODULE = model.module if distributed else model
            for name, para in MODULE.named_parameters():
                if para is not None:
                    for para_name in cfg.TENSORBOARD.PARAS:
                        if para_name in name:
                            tb_summary_writer.add_histogram(name.replace('.', '/'), para, iteration)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )


def get_angular_loss(weight):
    '''
    :param weight: parameter of model, out_features *　in_features
    :return: angular loss
    '''
    if weight.size(0) == 1:
        return 0.0

    # for convolution layers, flatten
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    # Dot product of normalized prototypes is cosine similarity.
    weight_ = F.normalize(weight, p=2, dim=1)
    product = torch.matmul(weight_, weight_.t())

    # Remove diagnonal from loss
    product_ = product - 2. * torch.diag(torch.diag(product))
    # Maxmize the minimum theta.
    loss = -torch.acos(product_.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

    return loss


def dma_loss(cosine):
    # pick the labeled  cos  theta
    max_cos = cosine.max(dim=1)[0]  # B
    min_theta = torch.acos(max_cos)  # B

    loss = 0.5 * min_theta.pow(2).mean()

    return loss
