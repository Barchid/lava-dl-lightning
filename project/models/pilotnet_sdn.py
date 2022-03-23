import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import lava.lib.dl.slayer as slayer


def event_rate_loss(x, max_rate=0.01):
    mean_event_rate = torch.mean(torch.abs(x))
    return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))


class Network(torch.nn.Module):
    def __init__(self, threshold=0.1, tau_grad=0.5, scale_grad=1., dropout=0.2):
        super(Network, self).__init__()

        sdnn_params = {  # sigma-delta neuron parameters
            'threshold': threshold,    # delta unit threshold
            'tau_grad': tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad': scale_grad,      # delta unit surrogate gradient scale parameter
            'requires_grad': True,   # trainable threshold
            'shared_param': True,   # layer wise threshold
            'activation': F.relu,  # activation function
        }
        sdnn_cnn_params = {  # conv layer has additional mean only batch norm
            **sdnn_params,                                 # copy all sdnn_params
            'norm': slayer.neuron.norm.MeanOnlyBatchNorm,  # mean only quantized batch normalizaton
        }
        sdnn_dense_params = {  # dense layers have additional dropout units enabled
            **sdnn_cnn_params,                        # copy all sdnn_cnn_params
            'dropout': slayer.neuron.Dropout(p=dropout),  # neuron dropout
        }

        self.blocks = torch.nn.ModuleList([  # sequential network blocks
            # delta encoding of the input
            slayer.block.sigma_delta.Input(sdnn_params),
            # convolution layers
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 3, 24, 3, padding=0,
                                          stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 24, 36, 3, padding=0,
                                          stride=2, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 36, 64, 3, padding=(1, 0),
                                          stride=(2, 1), weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Conv(sdnn_cnn_params, 64, 64, 3, padding=0,
                                          stride=1, weight_scale=2, weight_norm=True),
            # flatten layer
            slayer.block.sigma_delta.Flatten(),
            # dense layers
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 64 * 40, 100, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 100, 50, weight_scale=2, weight_norm=True),
            slayer.block.sigma_delta.Dense(sdnn_dense_params, 50, 10, weight_scale=2, weight_norm=True),
            # linear readout with sigma decoding of output
            slayer.block.sigma_delta.Output(sdnn_dense_params, 10, 1, weight_scale=2, weight_norm=True)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad
