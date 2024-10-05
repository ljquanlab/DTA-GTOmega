import torch
class BCELoss(torch.nn.Module):
    def __init__(self, pos_weight=1, reduction='mean'):
        super(BCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.neg_weight = 1 - self.pos_weight
        self.reduction = reduction

    def forward(self, probs, target):
        loss = - self.pos_weight * target * torch.log(probs) - \
               self.neg_weight * (1 - target) * torch.log(1 - probs)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss