import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    # def __init__(self):

    def forward(self, cls_preds, cls_targets, loc_preds, loc_targets):
        '''

        :param cls_preds: [batch_size, num_anchors, 20]
        :param cls_targets: [batch_size, num_anchors, 20]
        :param loc_preds: [batch_size, num_anchors, 4]
        :param loc_targets: [batch_size, num_anchors, 4]
        :return:
        '''
        alpha = 0.25
        gamma = 2.0
        batch_size = cls_preds.size(0)
        cls_losses = []
        loc_losses = []

        for i in range(batch_size):

            cls_pred = cls_preds[i, :, :]
            loc_pred = loc_preds[i, :, :]
            cls_target = cls_targets[i, :, :]
            loc_target = loc_targets[i, :, :]
            #cls_t_max, cls_t_argmax = torch.max(cls_target, dim=-1)
            #id = torch.argmax(cls_t_max)
           # print(cls_target[id, :])

            #cls_p_max, cls_p_argmax = torch.max(cls_pred, dim=-1)
            #print(cls_pred[id, :])

            cls_max, _ = torch.max(cls_target, dim=-1)
            pos_idx = torch.eq(cls_max, 1.)
            num_pos = pos_idx.sum()

            if loc_target.size(0) == 0:
                loc_losses.append(torch.tensor(0).float().cuda())
                cls_losses.append(torch.tensor(0).float().cuda())
                continue

            cls_pred = torch.clamp(cls_pred, 1e-4, 1.0 - 1e-4)

            alpha_factor = torch.ones(cls_target.size()).cuda() * alpha

            alpha_factor = torch.where(torch.eq(cls_target, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(cls_target, 1.), 1. - cls_pred, cls_pred)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(cls_target * torch.log(cls_pred) + (1.0 - cls_target) * torch.log(1.0 - cls_pred))
            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            cls_loss = torch.where(torch.ne(cls_target, -1.0), cls_loss, torch.zeros(cls_loss.size()).cuda())

            cls_losses.append(cls_loss.sum() / torch.clamp(num_pos.float(), min=1.0))

            # compute the loss for regression

            if pos_idx.sum() > 0:
                loc_target = loc_target[pos_idx, :]
                loc_pred = loc_pred[pos_idx, :]

                loc_diff = torch.abs(loc_pred - loc_target)

                loc_loss = torch.where(
                    torch.le(loc_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(loc_diff, 2),
                    loc_diff - 0.5 / 9.0
                )
                loc_losses.append(loc_loss.mean())
            else:
                loc_losses.append(torch.tensor(0).float().cuda())

        return torch.stack(cls_losses).mean(dim=0, keepdim=True), torch.stack(loc_losses).mean(dim=0, keepdim=True)


