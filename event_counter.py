import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class EventCounter(object):

    def __init__(self, net: nn.Module) -> None:
        if not hasattr(net, 'blocks'):
            raise ValueError('Network module has not "blocks" attribute. Cannot use Sparsity Loss.')

        self.net = net
        self.count = []

        self.hooks = []
        for block in self.net.blocks:

            if not hasattr(block, 'neuron'):  # only layers containing neurons
                continue

            hook = block.register_forward_hook(self.hook_get_events)
            self.hooks.append(hook)

    def hook_get_events(self, layer, input, output):
        events = output
        self.count.append(torch.sum(torch.abs((events[..., 1:]) > 0).to(events.dtype)).item())

    def get_count(self, input):
        result = self.count
        self.count = []  # reset for another forward pass
        count = torch.FloatTensor(result).reshape((1, -1))
        count = (count.flatten() / (input.shape[-1] - 1) / input.shape[0]).tolist()  # count skips first events
        return count
