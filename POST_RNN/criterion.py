import torch
from torch import nn
from torch.nn import functional as F

class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()

    def forward(self, x, label):
        B, L, C = x.size()
        loss_sum = torch.tensor(0).cuda().float()
        for i_batch in range(B):
            tmp_len = (label[i_batch] >= 0).sum()
            loss_sum += F.cross_entropy(x[i_batch, :tmp_len], label[i_batch, :tmp_len])
        loss_sum /= B

        return loss_sum
