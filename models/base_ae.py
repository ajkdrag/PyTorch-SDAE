import torch.nn as nn
import torch.optim as optim
from collections import namedtuple


class BaseAE:
    def __init__(self, network, device, hyps):
        self.network = network
        self.device = device
        self.hyps = hyps
        self.criterion = self.create_criterion()
        self.optimizer = self.create_optimizer()
        self.network.to(device)

    def create_criterion(self):
        return nn.MSELoss()

    def create_optimizer(self):
        return optim.Adam(self.network.parameters(), lr=self.hyps["adam_lr"])

    def compute_loss(self, model_op, model_ip):
        return self.criterion(model_op, model_ip)

    def yield_data(self, dataloader):
        for img, _ in dataloader:
            yield img

    def fit_one_cycle(self, dataloader, max_batches):
        Output = namedtuple("Output", ["loss", "reconstruction", "encoded"])
        for batch_id, img in enumerate(self.yield_data(dataloader)):
            if batch_id == max_batches:
                break
            img = img.to(self.device)
            encoded, reconstructed = self.network(img)
            loss = self.compute_loss(reconstructed, img)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return Output(loss=loss.item(), reconstruction=reconstructed, encoded=encoded)
