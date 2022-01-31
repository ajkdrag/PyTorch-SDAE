import torch.nn as nn
from functools import reduce
from operator import mul


class Autoencoder(nn.Module):
    def __init__(
        self,
        ip_shape,
        encoder_op_units,
        decoder_op_units,
        layer_activation_fn,
        output_activation_fn,
        symmetric=True,
    ):
        super().__init__()

        self.ip_shape = ip_shape
        flattened_size = reduce(mul, ip_shape)

        # encoder
        encoder_ip_units = [flattened_size] + encoder_op_units[:-1]
        self.encoder = Autoencoder.get_dense_block(
            encoder_ip_units,
            encoder_op_units,
            layer_activation_fn,
            prefix_layers=[nn.Flatten()],
        )

        # decoder
        if symmetric:
            decoder_ip_units = encoder_op_units[::-1]
        decoder_op_units = decoder_ip_units[1:] + [flattened_size]

        self.decoder = Autoencoder.get_dense_block(
            decoder_ip_units,
            decoder_op_units,
            layer_activation_fn,
            activate_last_layer=False,
        )

        # final activation
        self.output_activation_fn = output_activation_fn

    @staticmethod
    def get_dense_block(
        ip_units,
        op_units,
        activation_fn,
        activate_last_layer=True,
        prefix_layers=None,
        suffix_layers=None,
    ):
        if not prefix_layers:
            prefix_layers = []
        if not suffix_layers:
            suffix_layers = []

        layers = Autoencoder.make_dense(
            ip_units, op_units, activation_fn, activate_last_layer
        )

        dense_block = nn.Sequential(*prefix_layers, *layers, *suffix_layers)
        dense_block.apply(Autoencoder.init_weights)
        return dense_block

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def make_dense(
        list_ip_units, list_op_units, activation_fn, activate_last_layer=False
    ):
        layers = []
        num_layers = len(list_ip_units)
        for idx, pair in enumerate(zip(list_ip_units, list_op_units)):
            layers.append(nn.Linear(pair[0], pair[1]))
            if idx + (activate_last_layer ^ 1) < num_layers:
                layers.append(activation_fn)
        return layers

    def reconstruct(self, x):
        return self.output_activation_fn(self.decoder(x)).view(-1, *self.ip_shape)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.reconstruct(encoded)
        return encoded, decoded
