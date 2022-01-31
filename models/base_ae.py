import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
from utils.visualize import plot_one_list, plot_two_lists


Output = namedtuple("Output", ["loss"])


class BaseAE:
    def __init__(self, network, device, hyps):
        self.network = network
        self.device = device
        self.hyps = hyps
        self.create_criterion()
        self.create_optimizer()
        self.network.to(device)

    def create_criterion(self):
        self.criterion = nn.MSELoss()

    def create_optimizer(self):
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.hyps["adam_lr"])

    def compute_loss(self, pred, tgt):
        return self.criterion(pred, tgt)

    def yield_data(self, dataloader):
        for img, _ in dataloader:
            yield img

    def save_ae_outputs(self, og, recon):
        first_layer_weights = (
            list(self.network.encoder.children())[1].weight.cpu().detach()
        )
        plot_two_lists(og, recon, out="outputs/recon.png")
        plot_one_list(
            first_layer_weights.view(-1, *og.shape[-2:]), out="outputs/weights.png"
        )

    def fit_one_cycle(self, dataloader, max_batches, training=True, save_imgs=False):
        total_loss = 0
        if len(dataloader) > 0:
            for batch_id, og in enumerate(self.yield_data(dataloader)):
                img = og.to(self.device)
                _, reconstructed = self.network(img)
                loss = self.compute_loss(reconstructed, img)
                total_loss += loss.item()
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if batch_id + 1 == max_batches:
                    break
            if save_imgs:
                self.save_ae_outputs(og, reconstructed.cpu().detach())
            total_loss /= len(dataloader)
        return Output(loss=total_loss)
