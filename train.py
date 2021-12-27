import torch.nn as nn
from datasets.loader import get_dataloader
from networks.autoencoder import Autoencoder
from models.base_ae import BaseAE
from utils.general import get_device


class Trainer:
    def __init__(self, opts, hyps):
        self.opts = opts
        self.hyps = hyps
        # extract vars
        self.img_sz = self.opts["img_sz"]
        self.batch_sz = self.opts["batch_sz"]
        self.epochs = self.opts["epochs"]
        self.device = get_device()

    def setup_model(self):
        network = Autoencoder(
            (1, self.img_sz, self.img_sz),
            [256, 128],
            [],
            nn.ReLU(),
            nn.ReLU(),
            symmetric=True,
        )
        self.model = BaseAE(network, self.device, self.hyps)

    def setup_dataloader(self):
        self.train_loader, self.val_loader = get_dataloader(
            "mnist",
            self.opts["data"],
            self.batch_sz,
            (self.img_sz, self.img_sz),
            True,
            0.0,
            0,
            False,
        )

    def setup(self):
        self.setup_dataloader()
        self.setup_model()

    def run(self):
        max_batches = self.opts.get("max_batches", len(self.train_loader))
        for ep in range(self.epochs):
            op = self.model.fit_one_cycle(self.train_loader, max_batches=max_batches)
            print(op.loss, op.reconstruction.shape)