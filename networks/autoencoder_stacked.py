from torch import nn
from functools import reduce
from operator import mul
from networks.autoencoder import Autoencoder


class Stacked(nn.Module):
    """
    encoder_op_units = [[512, 256], [128, 64], [32, 16]]
    decoder_op_units = [[512, flattened_ip_sz], [128, 256], [32, 64]]
    """

    def __init__(
        self,
        ip_shape,
        list_encoder_op_units,
        list_decoder_op_units,
        layer_activation_fn,
        output_activation_fn,
        symmetric=True,
    ):
        super().__init__()
        if symmetric:
            list_decoder_op_units = []
            bottleneck_sz = reduce(mul, ip_shape)
            for encoder_ops in list_encoder_op_units:
                list_decoder_op_units.append(encoder_ops[1::-1] + [bottleneck_sz])
                bottleneck_sz = encoder_ops[-1]

        ae_ip_shape = ip_shape
        self.num_aes = len(list_encoder_op_units)
        self.aes = nn.ModuleList()
        for idx in range(self.num_aes):
            first_activation = layer_activation_fn if idx != 0 else output_activation_fn
            self.aes.append(
                Autoencoder(
                    ae_ip_shape,
                    list_encoder_op_units[idx],
                    list_decoder_op_units[idx],
                    layer_activation_fn,
                    first_activation,
                    True,
                ),
            )
            ae_ip_shape = (list_encoder_op_units[idx][-1],)

    def train(self):
        for ae in self.aes:
            ae.train()

    def eval(self):
        for ae in self.aes:
            ae.eval()

    def forward(self, x):
        ae_inputs = []
        ae_outputs = []
        for ae in self.aes:
            x = x.detach()
            ae_inputs.append(x)
            x, decoded = ae(x)
            ae_outputs.append(decoded)

        for idx in range(self.num_aes - 2, -1, -1):
            decoded = self.aes[idx].reconstruct(decoded)
        return ae_inputs, ae_outputs, decoded

