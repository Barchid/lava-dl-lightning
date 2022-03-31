from argparse import ArgumentParser
from lit_lavadl import LitLavaDL
import pytorch_lightning as pl
from project.dataset.nmnist import NMNISTDataset, augment
from project.models.pilotnet_sdn import Network
import os
import torch.nn.functional as F
from torchvision import transforms
import lava.lib.dl.slayer as slayer
from torch.utils.data import DataLoader


def program_args():
    parser = ArgumentParser()
    parser.add_argument('--mode', default="train", choices=["train", "test", "lr_find"],
                        help="Mode to run the program. Can be train, test, lr_find (finding learning rate).")
    parser.add_argument('--data_dir', default='data', help="Location of the dataset")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--ckpt_path', default=None,
                        help="Path of a checkpoint file. Defaults to None, meaning the training/testing will start from scratch.")
    parser.add_argument('--sampling_time', default=1, type=int, help="Number of bins")
    return parser


def create_module(dict_args) -> LitLavaDL:
    net = Network()
    module = LitLavaDL(
        net,
        error=slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum'),
        classifier=slayer.classifier.Rate.predict,
        **dict_args
    )

    return module


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # Get args
    # ------------
    parser = program_args()  # program args first
    parser = pl.Trainer.add_argparse_args(parser)  # Trainer args (gpus, etc...)
    parser = LitLavaDL.add_model_specific_args(parser)  # Model-related args
    args = parser.parse_args()  # get args
    dict_args = vars(args)

    # ------------
    # data
    # ------------
    train_set = NMNISTDataset(path='data', train=True, sampling_time=args.sample_time, transform=augment)
    val_set = NMNISTDataset(path='data', train=False, sampling_time=args.sample_time)
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    val_loader  = DataLoader(dataset=val_set , batch_size=args.batch_size, shuffle=True)

    # ------------
    # pytorch-lightning module
    # ------------
    module = create_module(dict_args)

    # ------------
    # trainer
    # ------------
    logger = pl.loggers.TensorBoardLogger(
        os.path.join(args.default_root_dir, 'logs'),
        name="logger"
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_total_loss",  # TODO: select the logged metric to monitor the checkpoint saving
        filename="model-{epoch:03d}-{val_total_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], logger=logger)

    # ------------
    # launch experiment
    # ------------
    if args.mode == "train":
        trainer.fit(module, train_loader, val_loader, ckpt_path=args.ckpt_path)
        # report results in a txt file
        report_path = os.path.join(args.default_root_dir, 'train_report.txt')
        report = open(report_path, 'a')

        # TODO: add any data you want to report here
        # here, we put the model's hyperparameters and the resulting val accuracy
        report.write(
            f"NMNIST {args.learning_rate} {args.lam} {trainer.checkpoint_callback.best_model_score}\n")

    elif args.mode == "test":
        trainer.validate(module, val_loader, ckpt_path=args.ckpt_path)

    elif args.mode == "lr_find":
        lr_finder = trainer.tuner.lr_find(module, train_loader, val_loader)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        print(f'SUGGESTION IS :', lr_finder.suggestion())
    else:
        raise NotImplementedError('No other option for --mode argument.')


if __name__ == '__main__':
    cli_main()
