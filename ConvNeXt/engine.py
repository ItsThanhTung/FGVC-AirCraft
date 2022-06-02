# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import utils
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        target_variant = targets[0].to(device, non_blocking=True)
        target_family = targets[1].to(device, non_blocking=True)
        # target_manufacturer = targets[2].to(device, non_blocking=True)
        ff_target_family = target_family.detach()
        with torch.no_grad():
            mask_family_tensor = []
            for i in range(target_variant.size()[0]):
                mask_family = np.ones((1, 224, 224)) * target_family[i].to("cuda").item()/70
                mask_family_tensor.append(mask_family)

            mask_family_tensor = torch.from_numpy(np.array(mask_family_tensor)).to('cuda', dtype=torch.float)

        # with torch.no_grad():
        #     padding_family_tensor = []
        #     # padding_manufacturer_tensor = []

        #     for i in range(target_variant.size()[0]):
        #         # padding_image_family = np.ones((1, 224, 224)) * target_family[i].to("cuda").item()
        #         # padding_image_manufacturer = np.ones((1, 224, 224)) * target_manufacturer[i].to("cuda").item()
        #         # padding_family_tensor.append(padding_image_family)
        #         # padding_manufacturer_tensor.append(padding_image_manufacturer)

        #     padding_family_tensor = torch.from_numpy(np.array(padding_family_tensor)).to('cuda', dtype=torch.float)
        #     # padding_manufacturer_tensor = torch.from_numpy(np.array(padding_manufacturer_tensor)).to('cuda', dtype=torch.float)

        #     samples = torch.cat((samples, padding_family_tensor), dim=1)


        if mixup_fn is not None:
            # samples, target_variant, target_family, target_manufacturer = mixup_fn(samples, \
            #                                                             target_variant, target_family, \
            #                                                             target_manufacturer)

            samples, target_variant, target_family = mixup_fn(samples, target_variant, target_family)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision          
            # output_variant, output_family, output_manufacturer = model(samples)
            # samples = torch.add(samples, mask_family_tensor).detach()
            output_variant, output_family = model(samples, mask_family_tensor, ff_target_family)
            loss1 = criterion(output_variant, target_variant)
            loss2 = criterion(output_family, target_family)
            # loss3 = criterion(output_manufacturer, target_manufacturer)
            loss = loss1*1 + loss2*0.3
        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})
            

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)

        target_variant = target[0].to(device, non_blocking=True)
        target_family = target[1].to(device, non_blocking=True)
        ff_target_family = target_family.detach()
        # target_manufacturer = target[2].to(device, non_blocking=True)
        with torch.no_grad():
            mask_family_tensor = []
            for i in range(target_variant.size()[0]):
                mask_family = np.ones((1, 224, 224)) * target_family[i].to("cuda").item()/70
                mask_family_tensor.append(mask_family)

            mask_family_tensor = torch.from_numpy(np.array(mask_family_tensor)).to('cuda', dtype=torch.float)

        # with torch.no_grad():
        #     padding_family_tensor = []
        #     # padding_manufacturer_tensor = []

        #     for i in range(target_variant.size()[0]):
        #         padding_image_family = np.ones((1, 224, 224)) * target_family[i].to("cuda").item()
        #         # padding_image_manufacturer = np.ones((1, 224, 224)) * target_manufacturer[i].to("cuda").item()
        #         padding_family_tensor.append(padding_image_family)
        #         # padding_manufacturer_tensor.append(padding_image_manufacturer)

        #     padding_family_tensor = torch.from_numpy(np.array(padding_family_tensor)).to('cuda', dtype=torch.float)
        #     # padding_manufacturer_tensor = torch.from_numpy(np.array(padding_manufacturer_tensor)).to('cuda', dtype=torch.float)

        #     images = torch.cat((images, padding_family_tensor), dim=1)



        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target_variant)
        else:
            # output_variant, output_family, output_manufacturer = model(images)
            # images = torch.add(images, mask_family_tensor).detach()
            output_variant, output_family = model(images, mask_family_tensor, ff_target_family)
            loss_variant = criterion(output_variant, target_variant)
            loss_family = criterion(output_family, target_family)
            # loss_manufacturer = criterion(output_manufacturer, target_manufacturer)


        acc1_variant, acc5_variant = accuracy(output_variant, target_variant, topk=(1, 5))
        acc1_family, acc5_family = accuracy(output_family, target_family, topk=(1, 5))
        # acc1_manufacturer, acc5_manufacturer = accuracy(output_manufacturer, target_manufacturer, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss_variant=loss_variant.item())
        metric_logger.meters['acc1_variant'].update(acc1_variant.item(), n=batch_size)
        metric_logger.meters['acc5_variant'].update(acc5_variant.item(), n=batch_size)

        metric_logger.update(loss_family=loss_family.item())
        metric_logger.meters['acc1_family'].update(acc1_family.item(), n=batch_size)
        metric_logger.meters['acc5_family'].update(acc5_family.item(), n=batch_size)

        # metric_logger.update(loss_manufacturer=loss_manufacturer.item())
        # metric_logger.meters['acc1_manufacturer'].update(acc1_manufacturer.item(), n=batch_size)
        # metric_logger.meters['acc5_manufacturer'].update(acc5_manufacturer.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* AccVariant@1 {top1.global_avg:.3f} AccVariant@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1_variant, top5=metric_logger.acc5_variant, losses=metric_logger.loss_variant))

    print('* AccFamily@1 {top1.global_avg:.3f} AccFamily@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1_family, top5=metric_logger.acc5_family, losses=metric_logger.loss_family))

    # print('* AccManufacturer@1 {top1.global_avg:.3f} AccManufacturer@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1_manufacturer, top5=metric_logger.acc5_manufacturer, losses=metric_logger.loss_manufacturer))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
