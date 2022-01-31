import torch
import torch.nn as nn
from datasets.loader import get_dataloader
from models.sparse_ae_kl import SparseAE
from models.denoising_ae import DenoisingAE
from models.stacked_ae import StackedDenoisingAE
from networks.autoencoder import Autoencoder
from networks.autoencoder_stacked import Stacked
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
            encoder_op_units=[256, 128],
            decoder_op_units=[],
            layer_activation_fn=nn.ReLU(),
            output_activation_fn=nn.Sigmoid(),
            symmetric=True,
        )
        # self.model = SparseAE(network, self.device, self.hyps)
#         network = Stacked(
#             ip_shape=(1, self.img_sz, self.img_sz),
#             list_encoder_op_units=[[256, 128], [64]],
#             list_decoder_op_units=[],
#             layer_activation_fn=nn.ReLU(),
#             output_activation_fn=nn.Sigmoid(),
#             symmetric=True,
#         )
        # self.model = StackedDenoisingAE(network, self.device, self.hyps)
        self.model = DenoisingAE(network, self.device, self.hyps)

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
        for ep in range(1, self.epochs + 1):
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
                    save_imgs=(True if ep % self.opts["save_freq"] == 0 else False),
                )

            print(
                f"Epoch: {ep}/{self.epochs} | Training loss: {train_op.loss} | Validation Loss: {val_op.loss}"
            )

