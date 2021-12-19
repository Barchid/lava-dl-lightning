import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import h5py

import lava.lib.dl.slayer as slayer


class Network(torch.nn.Module):
    def __init__(self, threshold=1.25, current_decay=0.25, voltage_decay=0.03, tau_grad=0.03, scale_grad=3., requires_grad=True, dropout=0.05):
        super(Network, self).__init__()

        neuron_params = {
            'threshold': threshold,
            'current_decay': current_decay,
            'voltage_decay': voltage_decay,
            'tau_grad': tau_grad,
            'scale_grad': scale_grad,
            'requires_grad': requires_grad,
        }

        neuron_params_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(dropout), }

        self.blocks = torch.nn.ModuleList([
            slayer.block.cuba.Dense(neuron_params_drop, 34 * 34 * 2, 512, weight_norm=True, delay=True),
            slayer.block.cuba.Dense(neuron_params_drop, 512, 512, weight_norm=True, delay=True),
            slayer.block.cuba.Dense(neuron_params, 512, 10, weight_norm=True),
        ])

    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)

        return spike

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))
