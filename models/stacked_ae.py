from importlib_metadata import requires
import torch
import numpy as np
from pathlib import Path
from torch import nn
from torch import optim
from skimage.util import random_noise
from collections import namedtuple
from functools import partial
from models.base_ae import BaseAE
from utils.visualize import plot_one_list, plot_two_lists


Output = namedtuple("Output", ["loss"])


class StackedDenoisingAE(BaseAE):
    def create_optimizer(self):
        self.optimizer = []
        for ae in self.network.aes:
            self.optimizer.append(optim.Adam(ae.parameters(), lr=self.hyps["adam_lr"],))

    def create_criterion(self):
        self.criterion = nn.MSELoss()

    def compute_loss(self, preds, tgts):
        """
        model_op: [(encoded_1, decoded_1), (encoded_2, decoded_2) ...]
        model_ip: og img
        """
        losses = [self.criterion(pred, tgt) for pred, tgt in zip(preds, tgts)]
        return losses

    def add_noise(self, inputs, mode="gaussian"):
        rand_noise = partial(random_noise, image=inputs, mode=mode, clip=True)
        if mode == "s&p":
            return torch.tensor(rand_noise(salt_vs_pepper=0.5), dtype=torch.float32,)
        elif mode == "speckle":
            return torch.tensor(rand_noise(mean=0, var=0.05), dtype=torch.float32,)
        else:
            return torch.tensor(rand_noise(mean=0, var=0.05), dtype=torch.float32,)

    def save_ae_outputs(self, og, recon):
        first_layer_weights = (
            list(self.network.aes[0].encoder.children())[1].weight.cpu().detach()
        )
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_two_lists(og, recon, out=str(out_dir / "recon.png"))
        plot_one_list(
            first_layer_weights.view(-1, *og.shape[-2:]), out=str(out_dir / "weights.png"))
    
    def fit_one_cycle(self, dataloader, max_batches, training=True, save_imgs=False):
        total_loss = [0] * self.network.num_aes
        if len(dataloader) > 0:
            for batch_id, og in enumerate(self.yield_data(dataloader)):
                noisy_og = self.add_noise(og, mode="s&p")
                og = og.to(self.device)
                img = noisy_og.to(self.device)
                ae_inputs, ae_outputs, reconstructed = self.network(img)
                ae_inputs[0] = og
                losses = self.compute_loss(ae_outputs, ae_inputs)
                total_loss = [
                    tloss + loss.item() for tloss, loss in zip(total_loss, losses)
                ]
                if training:
                    for idx, optimizer in enumerate(self.optimizer):
                        optimizer.zero_grad()
                        losses[idx].backward()
                        optimizer.step()
                if batch_id + 1 == max_batches:
                    break
            if save_imgs:
                self.save_ae_outputs(noisy_og, reconstructed.cpu().detach())
            total_loss = [tloss / og.shape[0] for tloss in total_loss]
        return Output(loss=total_loss)
