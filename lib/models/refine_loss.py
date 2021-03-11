import torch
import torch.nn as nn
import numpy as np


class Refine_loss(nn.Module):

    def __init__(self):
        super(Refine_loss, self).__init__()


    def forward(self, pred, gt):

        # loss_fn=torch.nn.L1Loss(reduction='mean')
        #
        # compensate_res_pred=pred[0]
        # resampled_pred=pred[1]
        # loss=loss_fn(compensate_res_pred,gt)+loss_fn(resampled_pred,gt)


        gt_mask=gt<0.0001
        pred0_mask=pred[0]<0.0001
        pred1_mask=pred[1]<0.0001

        mask=gt_mask|pred0_mask|pred1_mask

        masked_gt=gt.masked_fill(mask,value=torch.tensor(0.1))
        masked_pred0=pred[0].masked_fill(mask,value=torch.tensor(0.1))
        masked_pred1=pred[1].masked_fill(mask,value=torch.tensor(0.1))

        d1 = torch.log(masked_pred0) - torch.log(masked_gt)
        loss1= torch.sqrt((d1 ** 2).mean() - 0.85 * (d1.mean() ** 2)) * 10.0

        d2=torch.log(masked_pred1) - torch.log(masked_gt)
        loss2 = torch.sqrt((d2 ** 2).mean() - 0.85 * (d2.mean() ** 2)) * 10.0

        loss=loss1+loss2

        return loss





