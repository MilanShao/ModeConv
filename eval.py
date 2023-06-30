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


def getArgs(argv=None):
    parser = argparse.ArgumentParser(description="ModeConv")
    parser.add_argument("weights", metavar="Path",
                        help="path to pretrained weights, e.g. data/pretrained/luxemburg/ModeConvFast/GNN.statedict")
    parser.add_argument("--model", default="ModeConvFast", help="options: 'ModeConvFast', 'ModeConvLaplace', 'ChebConv', 'AGCRN', 'MtGNN'")
    parser.add_argument("--dataset", default="luxemburg", help="options: 'simulated_smart_bridge', 'luxemburg'")
    parser.add_argument('--batch-size', type=int, default=256, metavar="N")
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
    args.__dict__["lr"] = 0

    return args


if __name__ == "__main__":
    args = getArgs(sys.argv[1:])
    pl.seed_everything(args.seed)

    starttime = datetime.datetime.now()
    starttime = starttime.strftime("%H:%M:%S")

    model = select_model(args)
    model.load_state_dict(torch.load(args.weights))

    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=0,
        gpus=0 if args.no_cuda else 1,
    )

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
    out = {"args": vars(args), "Start time": starttime, "End time": endtime,
           "Last epoch duration": model.epoch_duration}
    with open(model.prefix + "/args.json", "w") as outfile:
        json.dump(out, outfile, indent=4, sort_keys=False)