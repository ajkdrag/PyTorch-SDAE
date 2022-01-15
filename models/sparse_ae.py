import torch
from torch import nn
from collections import namedtuple
from models.base_ae import BaseAE
from utils.general import get_children


Output = namedtuple("Output", ["loss"])


class SparseAE(BaseAE):
    def __init__(self, network, device, hyps, reg_param=0.01):
        super().__init__(network, device, hyps)
        self.reg_param = reg_param

    def sparse_loss(self, inputs):
        loss = 0
        values = inputs
        with torch.no_grad():
            for layer in get_children(self.network):
                values = layer(values)
                if isinstance(layer, (nn.ReLU, nn.Sigmoid)):
                    loss += torch.mean(torch.abs(values))
        return loss

    def compute_loss(self, model_op, model_ip):
        base_loss = self.criterion(model_op, model_ip)
        l1_loss = self.sparse_loss(model_ip)
        return base_loss + self.reg_param * l1_loss
