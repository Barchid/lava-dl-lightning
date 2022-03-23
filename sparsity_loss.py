import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def event_rate_loss(x, max_rate=0.01):
    mean_event_rate = torch.mean(torch.abs(x))
    return F.mse_loss(F.relu(mean_event_rate - max_rate), torch.zeros_like(mean_event_rate))


class Sparsity(object):

    def __init__(self, net: nn.Module, cost_function: function = event_rate_loss) -> None:
        if not hasattr(net, 'blocks'):
            raise ValueError('Network module has not "blocks" attribute. Cannot use Sparsity Loss.')

        self.net = net
        self.cost_function = cost_function
        self.total_loss = 0.

        self.hooks = []
        for block in self.net.blocks:

            if not hasattr(block, 'neuron'):  # only layers containing neurons
                continue

            hook = block.register_forward_hook(self.hook_get_spikes)
            self.hooks.append(hook)

    def hook_get_spikes(self, layer, input, output):
        spikes = output
        self.total_loss += self.cost_function(spikes)

    def get_sparsity_loss(self):
        result = self.total_loss
        self.total_loss = 0.  # reset for another forward pass
        return result
