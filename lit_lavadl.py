from argparse import ArgumentParser

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from sparsity_loss import Sparsity
from event_counter import EventCounter

import h5py


class LitLavaDL(pl.LightningModule):
    """
    Pytorch-Lightning module to train a model created with lava-dl and benefit from all functionalities.
    This can replace the default utils.Assistant from lava-dl.
    """

    def __init__(self, net: nn.Module, error, classifier=None, learning_rate: float = 0.001, lam: float = 0.3, count_log: bool = False, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['net', 'error', 'classifier', 'count_log'])

        self.net = net
        self.error = error
        self.classifier = classifier
        self.lam = lam
        self.sparsity = Sparsity(self.net) if lam is not None else None
        self.event_counter = EventCounter(self.net) if count_log else None

    def forward(self, x):
        events = self.model(x)

        if self.classifier is not None:
            classifications = self.classifier(events)
        else:
            classifications = None

        return events, classifications

    def training_step(self, batch, batch_idx):
        x, y = batch
        events, classifications = self(x)

        loss = self.error(events, y)
        self.log('train_loss', loss, on_epoch=True)

        if self.sparsity is not None:
            sparsity_loss = self.sparsity.get_sparsity_loss()
            self.log('train_sparsity_loss', sparsity_loss, on_epoch=True)

            loss += sparsity_loss
            self.log('train_total_loss', loss, on_epoch=True)

        if self.event_counter is not None:
            count = self.event_counter.get_count(x)
            print(count.shape, count)
            exit()
            self.log('train_count', count, on_epoch=True)

        if classifications is not None:
            acc = torchmetrics.functional.accuracy(classifications, y)
            self.log('train_acc', acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        events, classifications = self(x)

        loss = self.error(events, y)
        self.log('val_loss', loss, on_epoch=True)

        if self.sparsity is not None:
            sparsity_loss = self.sparsity.get_sparsity_loss()
            self.log('val_sparsity_loss', sparsity_loss, on_epoch=True)

            loss += self.lam * sparsity_loss
            self.log('val_total_loss', loss, on_epoch=True)

        if self.event_counter is not None:
            count = self.event_counter.get_count(x)
            print(count.shape, count)
            exit()
            self.log('val_count', count, on_epoch=True)

        if classifications is not None:
            acc = torchmetrics.functional.accuracy(classifications, y)
            self.log('val_acc', acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        events, classifications = self(x)

        loss = self.error(events, y)
        self.log('test_loss', loss, on_epoch=True)

        if self.sparsity is not None:
            sparsity_loss = self.sparsity.get_sparsity_loss()
            self.log('test_sparsity_loss', sparsity_loss, on_epoch=True)

            loss += sparsity_loss
            self.log('test_total_loss', loss, on_epoch=True)

        if self.event_counter is not None:
            count = self.event_counter.get_count(x)
            print(count.shape, count)
            exit()
            self.log('test_count', count, on_epoch=True)

        if classifications is not None:
            acc = torchmetrics.functional.accuracy(classifications, y)
            self.log('test_acc', acc, on_epoch=True)

    def export_hdf5(self, filename: str):
        """Exports the network model to hdf5 format.

        Args:
            filename (str): Filename of the exported model.
        """
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.net.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))

    def configure_optimizers(self):
        # TODO: change if you need another optimizer/scheduler
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--lam', type=float, default=0.3,
                            help="Lagrangian needed for the sparsity loss. If None, there is no sparsity loss (of course...)")
        parser.add_argument('--count_log', action="store_true", help="Activates the event count metric.")
        return parser
