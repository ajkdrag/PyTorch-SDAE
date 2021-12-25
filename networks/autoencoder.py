import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dims,
        encoder_op_units,
        decoder_op_units,
        layer_activation_fn,
        output_activation_fn,
        p=0.0,
        symmetric=True,
    ):
        super().__init__()

        # encoder
        encoder_ip_units = [input_dims] + encoder_op_units[:-1]
        self.encoder = Autoencoder.get_dense_block(
            encoder_ip_units, encoder_op_units, layer_activation_fn, p
        )

        # decoder
        if symmetric:
            decoder_ip_units = encoder_op_units[::-1]
        decoder_op_units = decoder_ip_units[1:] + [input_dims]

        self.decoder = Autoencoder.get_dense_block(
            decoder_ip_units, decoder_op_units, layer_activation_fn, 0.0, False
        )

        # final activation
        self.output_activation_fn = output_activation_fn

    @staticmethod
    def get_dense_block(
        ip_units, op_units, activation_fn, dropout_prob, activate_last_layer=True
    ):
        encoder_layers = [nn.Dropout(p=dropout_prob)] if dropout_prob > 0 else []
        encoder_layers += Autoencoder.make_dense(
            ip_units, op_units, activation_fn, activate_last_layer
        )

        dense_block = nn.Sequential(*encoder_layers)
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

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.output_activation_fn(self.decoder(encoded))
        return encoded, decoded
