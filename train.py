import torch
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
            ip_shape=(1, self.img_sz, self.img_sz),
            encoder_op_units=[128],
            decoder_op_units=[],
            layer_activation_fn=nn.ReLU(),
            output_activation_fn=nn.Sigmoid(),
            symmetric=True,
        )
        self.model = BaseAE(network, self.device, self.hyps)

    def setup_dataloader(self):
        self.train_loader, self.val_loader = get_dataloader(
            dataset_type="mnist",
            data_dir=self.opts["data"],
            batch_size=self.batch_sz,
            img_size=(self.img_sz, self.img_sz),
            shuffle=True,
            val_split=0.2,
            num_workers=0,
            pin_memory=False,
        )

    def setup(self):
        self.setup_dataloader()
        self.setup_model()

    def run(self):
        max_batches = self.opts.get("max_batches", len(self.train_loader))
        for ep in range(self.epochs):
            # training
            self.model.network.train()
            train_op = self.model.fit_one_cycle(
                self.train_loader, max_batches=max_batches
            )
            # validation
            self.model.network.eval()
            with torch.no_grad():
                val_op = self.model.fit_one_cycle(
                    self.val_loader,
                    max_batches=max_batches,
                    training=False,
                    save_imgs=(True if ep == self.epochs - 1 else False),
                )

            print(
                f"Epoch: {ep+1}/{self.epochs} | Training loss: {train_op.loss} | Validation Loss: {val_op.loss}"
            )

