import torch.nn as nn
from argparse import ArgumentParser
from datasets.loader import get_dataloader
from networks.autoencoder import Autoencoder
from models.base_ae import BaseAE


def parse_opt():
    parser = ArgumentParser()
    opt = parser.parse_args()
    return opt


def main(opt):
    print(opt)
    hyps = {"adam_lr": 3e-4, "img_sz": 24, "bs": 4}
    img_sz = hyps["img_sz"]
    autoencoder = Autoencoder(
        img_sz * img_sz, [256, 128], [], nn.ReLU(), nn.ReLU(), symmetric=True
    )
    train_loader, val_loader = get_dataloader(
        "mnist", "data", hyps["bs"], (img_sz, img_sz), True, 0.0, 0, False
    )
    single_batch = [next(iter(train_loader))]
    model = BaseAE(single_batch, autoencoder, "cpu", hyps)
    op = model.fit_one_cycle()
    print(op)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
