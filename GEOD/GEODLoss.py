import torch

from torch import nn 

class GEODLoss(nn.Module):
    def __init__(self, num_grid_cells) -> None:
        super(GEODLoss, self).__init__()
        self.num_grid_cells = num_grid_cells
        self.l_local = 1.25
    
    def forward(self, pred, target):
        return self.loss_cal(pred, target)
    
    def loss_cal(self, pred, target):

        # implement selection without thresholding so that gradient flow is perfect
        localization_loss_obj = torch.sum(target * ((pred-target)**2)) 

        negative_mask = torch.ones_like(target)
        target_neg = negative_mask - target
        localization_loss_noobj = torch.sum(target_neg * ((pred-target)**2))

        loss = self.l_local * localization_loss_obj + localization_loss_noobj

        return loss
