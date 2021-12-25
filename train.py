import torch.nn as nn
from argparse import ArgumentParser
from datasets.loader import get_dataloader 
from networks.fc_autoencoder import Autoencoder


def parse_opt():
    parser = ArgumentParser()
    opt = parser.parse_args()
    return opt


def main(opt):
    print(opt)
    # autoencoder = Autoencoder(784, [256, 128], [], nn.ReLU(), nn.ReLU(), symmetric=True)
    # train_loader, val_loader = get_dataloader("mnist", "data", 4, (24, 24), True, 0.0, 0, False)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
