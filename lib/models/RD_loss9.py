import torch
import torch.nn as nn
import numpy as np


class RD_loss9(nn.Module):

    def __init__(self, span=20, repeat=90):
        super(RD_loss9, self).__init__()
        self.span = span
        self.repeat = repeat

    def sub_block(self, tensor, pos):
        block = tensor[:, :, pos[0]:pos[1], pos[2]:pos[3]]

        return block

    def compute_rd(self, tensor,base_pos,rd_pos):

        base_block = self.sub_block(tensor, base_pos)
        rd_block=self.sub_block(tensor,rd_pos)

        return base_block-rd_block

    def forward(self, pred, gt):
        """
        compute the difference of relative depth map between predicted depth map and ground truth depth map
        :param pred: predicted depth map,B*1*H*W
        :param gt: ground truth depth map,B*1*H*W
        :return: difference of relative depth map between pred and gt
        """

        B, C, H, W = pred.shape

        # top,bottom,left,right
        base_block_pos = [self.span, H - self.span, self.span, W - self.span]

        loss = 0
        loss_fn = torch.nn.L1Loss(reduction='mean')

        pred_rd=[]
        gt_rd=[]

        for i in range(self.repeat):
            # the sequence of computing relative depth map:
            # top,bottom,left,right,top_left,top_right,bottom_left,bottom_right
            h_flag=np.random.uniform()
            h_offset=np.random.randint(1, self.span + 1)
            if h_flag>0.5:
                h_offset=-h_offset

            w_flag = np.random.uniform()
            w_offset = np.random.randint(1, self.span + 1)
            if w_flag > 0.5:
                w_offset = -w_offset

            # top,bottom,left,right
            rd_block_pos = [base_block_pos[0] +h_offset,
                            base_block_pos[1] +h_offset,
                            base_block_pos[2] +w_offset,
                            base_block_pos[3] +w_offset]

            pred_rd.append(self.compute_rd(pred,base_block_pos,rd_block_pos))
            gt_rd.append(self.compute_rd(gt,base_block_pos,rd_block_pos))


        pred_rd=torch.cat(pred_rd,1).permute(0,2,3,1)
        gt_rd=torch.cat(gt_rd,1).permute(0,2,3,1)

        pred_norm = torch.norm(pred_rd, 2, dim=3, keepdim=True)

        gt_norm = torch.norm(gt_rd, 2, dim=3, keepdim=True)

        pred_mask = pred_norm == 0
        gt_mask = gt_norm == 0

        pred_norm = pred_norm.masked_fill(pred_mask, value=1.0)
        gt_norm = gt_norm.masked_fill(gt_mask, value=1.0)

        pred_rd = pred_rd / pred_norm
        gt_rd = gt_rd / gt_norm

        loss =  loss+loss_fn(pred_rd, gt_rd)

        return loss





