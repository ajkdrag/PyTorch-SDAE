import torch
import torch.nn.functional as F
from torch import nn
from collections import namedtuple
from models.base_ae import BaseAE
from utils.general import get_children, kl_divergence


Output = namedtuple("Output", ["loss"])


class SparseAE(BaseAE):
    def __init__(self, network, device, hyps, reg_param=0.01):
        super().__init__(network, device, hyps)
        self.reg_param = reg_param
        self.rho = 0.01

    def sparse_loss(self, inputs):
        loss = 0
        values = inputs
        for layer in get_children(self.network):
            values = layer(values)
            if isinstance(layer, nn.Linear):
                rho_hat = torch.mean(values, dim=0, keepdim=True)
                rho = torch.FloatTensor(
                    [self.rho] * rho_hat.shape[-1], device=self.device
                ).unsqueeze(0)
                loss += kl_divergence(F.softmax(rho, dim=1), F.softmax(rho_hat, dim=1))
        return loss

    def compute_loss(self, model_op, model_ip):
        base_loss = self.criterion(model_op, model_ip)
        kl_loss = self.sparse_loss(model_ip)
        return base_loss + self.reg_param * kl_loss 
