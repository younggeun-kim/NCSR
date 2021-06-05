import torch
from torch import nn as nn

from models.modules import thops
from utils.util import opt_get


class StdConditionalLayer(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.scaleLayer = nn.Linear(1, dim_out)
        self.shiftLayer = nn.Linear(1, dim_out)
        
        m = self.scaleLayer
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight.data) 
            m.bias.data.fill_(0.01)

        m = self.shiftLayer
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight.data) 
            m.bias.data.fill_(0.01)
            
    def forward(self, input: torch.Tensor, logdet=None, reverse=False, std=None):
        if not reverse:
            z = input
            b = z.size(0)
            # Std Conditional
            scaleFt, shiftFt = torch.sigmoid(self.scaleLayer(std.view(-1,1)) + 2.).view(b, -1, 1, 1), self.shiftLayer(std.view(-1,1)).view(b, -1, 1, 1)
            z = z + shiftFt
            z = z * scaleFt
            logdet = logdet + self.get_logdet(scaleFt)
            
            output = z
        else:
            z = input
            b = z.size(0)
            # Std Conditional
            scaleFt, shiftFt = torch.sigmoid(self.scaleLayer(std.view(-1,1)) + 2.).view(b, -1, 1, 1), self.shiftLayer(std.view(-1,1)).view(b, -1, 1, 1)
            z = z / scaleFt
            z = z - shiftFt
            logdet = logdet - self.get_logdet(scaleFt)

            output = z
        return output, logdet
        
    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1])

