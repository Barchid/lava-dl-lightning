from argparse import ArgumentParser
import os
from pytorch_lightning.core import datamodule

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from project.models.pilotnet_sdn import Network


class LitPilotNet(pl.LightningModule):
    def __init__(self, threshold=0.1, tau_grad=0.5, scale_grad=1., dropout=0.2, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = Network(
            threshold=self.hparams.threshold,
            tau_grad=self.hparams.tau_grad,
            scale_grad=self.hparams.scale_grad,
            dropout=self.hparams.dropout
        )

        self.criterion = lambda output, target: F.mse_loss(output.flatten(), target.flatten())

    def forward(self, x):
        spikes = self.model(x)
        preds = self.classifier(spikes)
        return spikes, preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        spikes, preds = self(x)

        loss = self.criterion(spikes, y)
        self.log('train_loss', loss, on_epoch=True)

        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('train_acc', acc, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spikes, preds = self(x)

        loss = self.criterion(spikes, y)
        self.log('val_loss', loss, on_epoch=True)

        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('val_acc', acc, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spikes, preds = self(x)

        loss = self.criterion(spikes, y)
        self.log('test_loss', loss, on_epoch=True)

        acc = torchmetrics.functional.accuracy(preds, y)
        self.log('test_acc', acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--threshold', type=float, default=0.1)
        parser.add_argument('--tau_grad', type=float, default=0.5)
        parser.add_argument('--scale_grad', type=float, default=1.)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--lam', type=float, default=None, help="Lagrangian needed for the sparsity loss")
        return parser


def program_args():
    parser = ArgumentParser()
    parser.add_argument('--mode', default="train",
                        choices=["train", "test", "lr_find"], help="Mode to run the program. Can be train, test or lr_finder.")
    parser.add_argument('--data_dir', default='data/nmnist', help="Location of the dataset to train")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--ckpt_path', default=None,
                        help="Path of a checkpoint file. Defaults to None, meaning the training/testing will start from scratch.")
    return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # Get args
    # ------------
    parser = program_args()  # program args first
    parser = pl.Trainer.add_argparse_args(parser)  # Trainer args (gpus, etc...)
    parser = LitNMNIST.add_model_specific_args(parser)  # Model-related args
    args = parser.parse_args()  # get args
    dict_args = vars(args)

    # ------------
    # data
    # ------------
    data_module = NMNISTDataModule(
        **dict_args
    )

    # ------------
    # pytorch-lightning module
    # ------------
    module = LitNMNIST(
        **dict_args
    )

    # ------------
    # trainer
    # ------------
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="model-{epoch:03d}-{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])

    # ------------
    # launch experiment
    # ------------
    if args.mode == "train":
        trainer.fit(module, datamodule=data_module, ckpt_path=args.ckpt_path)
        # report results in a txt file
        report_path = os.path.join(args.default_root_dir, 'train_report.txt')
        report = open(report_path, 'a')

        # TODO: add any data you want to report here
        # here, we put the model's hyperparameters and the resulting val accuracy
        report.write(
            f"NMNIST-SLAYER {args.learning_rate} {args.loss_module} {trainer.checkpoint_callback.best_model_score}\n")

    elif args.mode == "test":
        trainer.validate(module, datamodule=datamodule, ckpt_path=args.ckpt_path)

    elif args.mode == "lr_find":
        lr_finder = trainer.tuner.lr_find(module, datamodule=datamodule)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        print(f'SUGGESTION IS :', lr_finder.suggestion())

    else:
        raise NotImplementedError('No other option for --mode argument.')


if __name__ == '__main__':
    cli_main()
