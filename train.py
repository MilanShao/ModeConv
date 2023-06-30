import datetime
import json
import sys
import os
import torch
import torch_geometric as pyg
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from ModeConvModel.models.select_model import select_model
from ModeConvModel.dataset import SimulatedSmartBridgeDataset, LuxemburgDataset
import argparse


def train(data_module, args):
    starttime = datetime.datetime.now()
    starttime = starttime.strftime("%H:%M:%S")

    model = select_model(args)
    auto_lr = True if args.lr == "auto" else False

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=args.epochs,
        gpus=0 if args.no_cuda else 1,
        auto_lr_find=auto_lr                         # run learning rate finder, results override hparams.learning_rate
    )

    # call tune to find the batch_size and to optimize lr
    trainer.tune(model, data_module)
    data_module.kwargs["batch_size"] = model.batch_size
    trainer.fit(model, data_module)
    torch.save(model.state_dict(), model.prefix + "/GNN.statedict")
    del data_module

    trainendtime = datetime.datetime.now()
    trainendtime = trainendtime.strftime("%H:%M:%S")
    print("Current Time =", trainendtime)
    print()

    if args.dataset == "simulated_smart_bridge":
        val_ds = SimulatedSmartBridgeDataset("./processed_simulated_smart_bridge/", mode="val")
    elif args.dataset == "luxemburg":
        val_ds = LuxemburgDataset("./processed_lux", mode="val")
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    trainer.validate(model, val_dl)
    del val_ds, val_dl

    if args.dataset == "simulated_smart_bridge":
        test_ds = SimulatedSmartBridgeDataset("./processed_simulated_smart_bridge/", mode="test")
    elif args.dataset == "luxemburg":
        test_ds = LuxemburgDataset("./processed_lux", mode="test")

    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    trainer.test(model, test_dl)

    endtime = datetime.datetime.now()
    endtime = endtime.strftime("%H:%M:%S")
    out = {"args": vars(args), "Start time": starttime, "Train end time": trainendtime, "End time": endtime,
           "Last epoch duration": model.epoch_duration}
    with open(model.prefix + "/args.json", "w") as outfile:
        json.dump(out, outfile, indent=4, sort_keys=False)


def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="ModeConv")
    parser.add_argument("--model", default="ModeConvFast", help="options: 'ModeConvFast', 'ModeConvLaplace', 'ChebConv', 'AGCRN', 'MtGNN'")
    parser.add_argument("--dataset", default="luxemburg", help="options: 'simulated_smart_bridge', 'luxemburg'")
    parser.add_argument("--epochs", type=int, default=50, metavar="N")
    parser.add_argument('--batch-size', type=int, default=256, metavar="N")
    parser.add_argument("--lr", default="auto", metavar="lr",  # 1e-4
                        help="initial learning rate for optimizer e.g.: 1e-4 | 'auto'")
    parser.add_argument('--num-layer', type=int, default=3, metavar="N")
    parser.add_argument('--decoder', default="custom", help="options: 'linear' for linear layer decoder;"
                                                                     "'custom': to use ModeConv/ChebConv/etc. layers in decoder")
    parser.add_argument('--hidden-dim', type=int, default=8, metavar="N")
    parser.add_argument('--bottleneck', type=int, default=2, metavar="N")
    parser.add_argument("--no-maha-threshold", action="store_true", default=True,
                        help="mahalanobis threshold calculation/evaluation is very slow; disabling saves 30min+ in val and test on luxemburg dataset")
    parser.add_argument('--seed', type=int, default=3407, metavar="N")
    parser.add_argument("--no-cuda", action="store_true", default=False)

    args = parser.parse_args(argv)

    return args


if __name__ == "__main__":
    args = getArgs(sys.argv[1:])

    pl.seed_everything(args.seed)
    if args.dataset == "simulated_smart_bridge":
        train_ds = SimulatedSmartBridgeDataset("./processed_simulated_smart_bridge/", mode="train")
    elif args.dataset == "luxemburg":
        train_ds = LuxemburgDataset("./processed_lux", mode="train")
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not found; Choices: simulated_smart_bridge, luxemburg")

    data_module = pyg.data.LightningDataset(
        train_dataset=train_ds,
        # val_dataset=val_ds,
        # test_dataset=test_ds,
        batch_size=args.batch_size,
        num_workers=0
    )

    train(data_module=data_module, args=args)
