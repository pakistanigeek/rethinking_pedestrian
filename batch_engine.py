import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tools.utils import AverageMeter, to_scalar, time_str


def batch_trainer(epoch, model, train_loader, criterion, optimizer):
    model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()

    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []
    ft_lr = optimizer.param_groups[0]['lr'] 
    lr = optimizer.param_groups[1]['lr']

    for step, (imgs, gt_label, imgname) in enumerate(train_loader):

        batch_time = time.time()
        if torch.cuda.is_available():
            imgs, gt_label = imgs.cuda(), gt_label.cuda()
        train_logits = model(imgs, gt_label)
        # train_loss = criterion(train_logits, gt_label)
        loss_list = []
        # deep supervision
        for k in range(len(train_logits)):
            out = train_logits[k]
            loss_list.append(criterion(out, gt_label))

        train_loss = sum(loss_list)
        # maximum voting
        train_logits = torch.max(train_logits[0], train_logits[1])

        train_loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=12.0)  # make larger learning rate works
        optimizer.step()
        optimizer.zero_grad()
        loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logits)
        preds_probs.append(train_probs.detach().cpu().numpy())

        log_interval = 20
        if (step + 1) % log_interval == 0 or (step + 1) % len(train_loader) == 0:
            print(f'{time_str()}, Step {step}/{batch_num} in Ep {epoch}, {time.time() - batch_time:.2f}s ',
                  f'train_loss:{loss_meter.val:.4f}')

    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    print(f'Epoch {epoch}, FTLR {ft_lr},LR {lr}, Train_Time {time.time() - epoch_time:.2f}s, Loss: {loss_meter.avg:.4f}')

    return train_loss, gt_label, preds_probs


# @torch.no_grad()
def valid_trainer(model, valid_loader, criterion):
    model.eval()
    loss_meter = AverageMeter()

    preds_probs = []
    gt_list = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(tqdm(valid_loader)):
            if torch.cuda.is_available():
                imgs, gt_label = imgs.cuda(), gt_label.cuda()
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits = model(imgs)
            # valid_loss = criterion(valid_logits, gt_label)
            loss_list = []
            # deep supervision
            for k in range(len(valid_logits)):
                out = valid_logits[k]
                loss_list.append(criterion(out, gt_label))

            valid_loss = sum(loss_list)
            valid_logits = torch.max(valid_logits[0], valid_logits[1])

            valid_probs = torch.sigmoid(valid_logits)
            preds_probs.append(valid_probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs
