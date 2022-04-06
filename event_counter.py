import torch
import torch.nn as nn
import numpy as np


class EventCounter(object):

    def __init__(self, net: nn.Module) -> None:
        if not hasattr(net, 'blocks'):
            raise ValueError(
                'Network module has no "blocks" attribute containing the network layers. Cannot use this event counter tool.')

        self.net = net
        self.count = []  # list containing the event count for the current forward pass
        self.total_counts = []  # list containing the event counts for the whole epoch

        self.hooks = []
        for block in self.net.blocks:

            if not hasattr(block, 'neuron'):  # only layers containing neurons
                continue

            hook = block.register_forward_hook(self.hook_get_events)
            self.hooks.append(hook)

    def hook_get_events(self, layer, input, output):
        events = output.clone().detach().cpu()  # put the events tensor in CPU to avoid VRAM consumption
        self.count.append(torch.sum(torch.abs((events[..., 1:]) > 0).to(events.dtype)).item())

    def compute_count_forward(self, input):
        result = self.count
        self.count = []  # reset for another forward pass
        count = torch.FloatTensor(result).reshape((1, -1))  # per-layer total sum of events
        count = (count.flatten() / (input.shape[-1] - 1) / input.shape[0]).tolist()  # count skips first events
        self.total_counts.append(count)
        print(count)

    def get_ops_comparison(self):
        result = self.total_counts
        self.total_counts = []  # reset for another epoch
        counts = np.mean(counts, axis=0)  # mean over all counts
        
        return self.compare_ops(counts)
        

    def compare_ops(self, counts):
        shapes = [b.shape for b in self.net.blocks if hasattr(b, 'neuron')]

        # synops calculation
        sdnn_synops = []
        ann_synops = []
        for l in range(1, len(self.net.blocks)):
            if hasattr(self.net.blocks[l], 'neuron') is False:
                break
            conv_synops = (  # ignoring padding
                counts[l - 1] *
                self.net.blocks[l].synapse.out_channels *
                np.prod(self.net.blocks[l].synapse.kernel_size) /
                np.prod(self.net.blocks[l].synapse.stride)
            )
            sdnn_synops.append(conv_synops)
            ann_synops.append(conv_synops * np.prod(self.net.blocks[l - 1].shape) / counts[l - 1])
            # ann_synops.append(conv_synops*np.prod(self.net.blocks[l-1].shape)/counts[l-1]*np.prod(self.net.blocks[l].synapse.stride))

        for l in range(l + 1, len(self.net.blocks)):
            fc_synops = counts[l - 2] * self.net.blocks[l].synapse.out_channels
            sdnn_synops.append(fc_synops)
            ann_synops.append(fc_synops * np.prod(self.net.blocks[l - 1].shape) / counts[l - 2])

        # event and synops comparison
        total_events = np.sum(counts)
        total_synops = np.sum(sdnn_synops)
        total_ann_activs = np.sum([np.prod(s) for s in shapes])
        total_ann_synops = np.sum(ann_synops)
        total_neurons = np.sum([np.prod(s) for s in shapes])
        steps_per_inference = 1

        # print(f'|{"-"*77}|')
        # print('|', ' ' * 23, '|          SDNN           |           ANN           |')
        # print(f'|{"-"*77}|')
        # print('|', ' ' * 7, f'|     Shape     |  Events  |    Synops    | Activations|    MACs    |')
        # print(f'|{"-"*77}|')
        # for l in range(len(counts)):
        #     print(f'| layer-{l} | ', end='')
        #     if len(shapes[l]) == 3:
        #         z, y, x = shapes[l]
        #     elif len(shapes[l]) == 1:
        #         z = shapes[l][0]
        #         y = x = 1
        #     print(f'({x:-3d},{y:-3d},{z:-3d}) | {counts[l]:8.2f} | ', end='')
        #     if l == 0:
        #         print(f'{" "*12} | {np.prod(shapes[l]):-10.0f} | {" "*10} |')
        #     else:
        #         print(f'{sdnn_synops[l-1]:12.2f} | {np.prod(shapes[l]):10.0f} | {ann_synops[l-1]:10.0f} |')
        # print(f'|{"-"*77}|')
        # print(
        #     f'|  Total  | {" "*13} | {total_events:8.2f} | {total_synops:12.2f} | {total_ann_activs:10.0f} | {total_ann_synops:10.0f} |')
        # print(f'|{"-"*77}|')

        # print('\n')
        # print(f'MSE            : {mse:.5} sq. radians')
        # print(f'Total neurons  : {total_neurons}')
        # print(f'Events sparsity: {total_ann_activs/total_events:5.2f}x')
        # print(f'Synops sparsity: {total_ann_synops/total_synops:5.2f}x')
        
        return {
            'total_events': total_events,
            'event_sparsity': total_ann_activs/total_events,
            'synops_sparsity':total_ann_synops/total_synops,
            'total_neurons': total_neurons
        }
