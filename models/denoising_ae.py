import torch
from skimage.util import random_noise
from collections import namedtuple
from functools import partial
from models.base_ae import BaseAE


Output = namedtuple("Output", ["loss"])


class DenoisingAE(BaseAE):
    def add_noise(self, inputs, mode="gaussian"):
        rand_noise = partial(random_noise, image=inputs, mode=mode, clip=True)
        if mode == "s&p":
            return torch.tensor(rand_noise(salt_vs_pepper=0.5), dtype=torch.float32,)
        elif mode == "speckle":
            return torch.tensor(rand_noise(mean=0, var=0.05), dtype=torch.float32,)
        else:
            return torch.tensor(rand_noise(mean=0, var=0.05), dtype=torch.float32,)

    def fit_one_cycle(self, dataloader, max_batches, training=True, save_imgs=False):
        total_loss = 0
        if len(dataloader) > 0:
            for batch_id, og in enumerate(self.yield_data(dataloader)):
                noisy_og = self.add_noise(og, mode="s&p")
                img = noisy_og.to(self.device)
                _, reconstructed = self.network(img)
                loss = self.compute_loss(reconstructed, og)
                total_loss += loss.item()
                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                if batch_id + 1 == max_batches:
                    break
            if save_imgs:
                self.save_ae_outputs(noisy_og, reconstructed.cpu().detach())
            total_loss /= len(dataloader)
        return Output(loss=total_loss)
