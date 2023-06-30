import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.nn import Linear
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch_geometric.nn.dense.linear import Linear

from ModeConvModel.models.utils import compute_thresholds, compute_maha_threshold, generate_date_prefix, \
    compute_and_save_metrics


class ChebGNNEncoder(torch.nn.Module):
    def __init__(self, num_sensors, args):
        super().__init__()
        if args.num_layer <= 1:
            self.conv_layers = nn.ModuleList([ChebConv(14 * 25, args.bottleneck, K=5)])
            self.lin_layers = nn.ModuleList([Linear(14 * 25, args.bottleneck)])
        else:
            self.conv_layers = nn.ModuleList([ChebConv(14 * 25, args.hidden_dim, K=5)])
            self.lin_layers = nn.ModuleList([Linear(14 * 25, args.hidden_dim)])
            for i in range(args.num_layer - 2):
                self.conv_layers.append(ChebConv(args.hidden_dim, args.hidden_dim, K=5))
                self.lin_layers.append(Linear(args.hidden_dim, args.hidden_dim))

            self.conv_layers.append(ChebConv(args.hidden_dim, args.bottleneck, K=5))
            self.lin_layers.append(Linear(args.hidden_dim, args.bottleneck))

    def forward(self, x, edge_index):
        for conv, lin in zip(self.conv_layers, self.lin_layers):
            x = conv(x, edge_index) + lin(x)
            x = x.relu()
        return x


class ChebEdgeDecoder(torch.nn.Module):
    def __init__(self, num_sensors, args):
        super().__init__()
        self.args = args
        if args.decoder == "linear":
            if args.num_layer <= 1:
                self.lin_layers = nn.ModuleList([Linear(args.bottleneck, 14 * 25)])
            else:
                self.lin_layers = nn.ModuleList([Linear(args.bottleneck, args.hidden_dim)])
                for i in range(args.num_layer - 2):
                    self.lin_layers.append(Linear(args.hidden_dim, args.hidden_dim))
                self.lin_layers.append(Linear(args.hidden_dim, 14 * 25))
        elif args.decoder == "custom":
            if args.num_layer <= 1:
                self.conv_layers = nn.ModuleList([ChebConv(args.bottleneck, 14 * 25, K=5, num_sensors=num_sensors)])
                self.lin_layers = nn.ModuleList([Linear(args.bottleneck, 14 * 25)])
            else:
                self.conv_layers = nn.ModuleList(
                    [ChebConv(args.bottleneck, args.hidden_dim, K=5, num_sensors=num_sensors)])
                self.lin_layers = nn.ModuleList([Linear(args.bottleneck, args.hidden_dim)])
                for i in range(args.num_layer - 2):
                    self.conv_layers.append(
                        ChebConv(args.hidden_dim, args.hidden_dim, K=5, num_sensors=num_sensors))
                    self.lin_layers.append(Linear(args.hidden_dim, args.hidden_dim))

                self.conv_layers.append(ChebConv(args.hidden_dim, 14 * 25, K=5, num_sensors=num_sensors))
                self.lin_layers.append(Linear(args.hidden_dim, 14 * 25))
        else:
            raise NotImplementedError(f"Decoder {args.decoder} not implemented")

    def forward(self, z, edge_index):
        if self.args.decoder == "linear":
            for lin in self.lin_layers[:-1]:
                z = lin(z).relu()
            z = self.lin_layers[-1](z)
        elif self.args.decoder == "custom":
            for conv, lin in zip(self.conv_layers[:-1], self.lin_layers[:-1]):
                z = conv(z, edge_index) + lin(z)
                z = z.relu()
            z = self.conv_layers[-1](z, edge_index) + self.lin_layers[-1](z)
        return z.view(-1)


class ChebGNNModel(torch.nn.Module):
    def __init__(self, num_sensors, args):
        super().__init__()
        self.encoder = ChebGNNEncoder(num_sensors, args)
        self.decoder = ChebEdgeDecoder(num_sensors, args)

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        z = self.decoder(z, edge_index)
        return z


class LuxChebGNNModelPL(pl.LightningModule):
    def __init__(self, num_sensors: int, args) -> None:
        super().__init__()
        self.model = ChebGNNModel(num_sensors, args)
        self.num_sensors = num_sensors
        self.save_hyperparameters()
        self.learning_rate = 1e-3 if args.lr == "auto" else float(args.lr)
        self.batch_size = args.batch_size
        prefix = generate_date_prefix()
        random.seed(args.seed)
        self.args = args
        self.prefix = os.path.dirname(os.path.abspath(__file__)) + f'/../../results/{args.dataset}/{args.model}/{prefix}'
        Path(self.prefix).mkdir(parents=True, exist_ok=True)
        self.epoch_duration = 0
        self.val_losses = []
        self.val_logits = []
        self.val_labels = []
        self.test_data = []
        self.test_scores = []
        self.test_maha_scores = []
        self.test_logits = []
        self.test_labels = []
        self.dev = "cpu" if args.no_cuda else "cuda"
        with open('processed_lux/processed/metadata.json', 'r') as fp:
            stats = json.load(fp)
        self.min = torch.tensor(np.concatenate((np.tile(stats["voltage"]["min"], 25), np.tile(stats["temp"]["min"], 25), np.tile(stats["disp"]["min"], 25), np.tile(stats["acceleration"]["min"], 25))), device=self.dev, dtype=self.dtype)
        self.max = torch.tensor(np.concatenate((np.tile(stats["voltage"]["max"], 25), np.tile(stats["temp"]["max"], 25), np.tile(stats["disp"]["max"], 25), np.tile(stats["acceleration"]["max"], 25))), device=self.dev, dtype=self.dtype)
        self.mean = torch.tensor(np.concatenate((np.tile(stats["voltage"]["mean"], 25), np.tile(stats["temp"]["mean"], 25), np.tile(stats["disp"]["mean"], 25), np.tile(stats["acceleration"]["mean"], 25))), device=self.dev, dtype=self.dtype)
        self.std = torch.tensor(np.concatenate((np.tile(stats["voltage"]["std"], 25), np.tile(stats["temp"]["std"], 25), np.tile(stats["disp"]["std"], 25), np.tile(stats["acceleration"]["std"], 25))), device=self.dev, dtype=self.dtype)

    def forward(self, x, edge_index):
        z = self.model(x, edge_index)
        return z

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def process_batch(self, batch, batch_idx):
        x, edge_index, acceleration, disp = batch.x, batch.edge_index, batch.acc, batch.disp
        if batch_idx == 0:
            self.epoch_start_time = time.time()
        acceleration = acceleration.view((batch.num_graphs, 5, 5, acceleration.shape[-1]))

        num_sensors = acceleration.shape[-1]

        e = np.vstack((np.repeat(np.arange(num_sensors), num_sensors), np.tile(np.arange(num_sensors), num_sensors)))
        e = e[:, np.where(e[0] != e[1])[0]]
        add = np.arange(0, batch.num_graphs)
        add = np.repeat(add, e.shape[1])
        add *= num_sensors
        e1 = np.tile(e, batch.num_graphs)
        e2 = e1 + add
        edge_index = torch.tensor(e2, device=self.dev, dtype=torch.int64)

        disp = disp.view((batch.num_graphs, 1, -1, disp.shape[-1])).repeat((1, num_sensors, 1, 1))

        # reshape and add acceleration to x
        x = x.view((batch.num_graphs, 1, -1, x.shape[-1]))
        x = x.repeat(1, num_sensors, 1, 1)
        x = torch.concatenate((x.view((x.shape[0], x.shape[1], -1)), disp.view((x.shape[0], x.shape[1], -1)),
                               acceleration.permute(0, 3, 1, 2).reshape(acceleration.shape[0], acceleration.shape[3],
                                                                        -1)), dim=-1)
        x = x.view((-1, x.shape[-1]))

        # x = (x - self.mean) / self.std
        x = (x - self.min) / (self.max - self.min)

        return x, edge_index

    def training_step(self, train_batch: Data, batch_idx):
        x, edge_index = self.process_batch(train_batch, batch_idx)

        y_hat = self.forward(x, edge_index)

        losses = torch.nn.MSELoss()(y_hat, x.view(-1))

        return losses

    def training_epoch_end(self, outputs) -> None:
        self.epoch_duration = time.time() - self.epoch_start_time
        self.log("loss/train", float(np.mean([x["loss"].cpu() for x in outputs])))

    def validation_step(self, val_batch: Data, batch_idx):
        x, edge_index = self.process_batch(val_batch, batch_idx)

        y_hat = self.forward(x, edge_index)

        batchsize = len(val_batch.y)
        losses = torch.nn.MSELoss(reduction="none")(y_hat, x.view(-1)).view(batchsize, -1).mean(axis=1)
        self.val_losses.append(losses)

        y = val_batch.y
        self.val_labels.append(y)
        self.val_logits.append(y_hat.cpu())
        return losses.mean()

    def validation_epoch_end(self, outputs) -> None:
        self.val_losses = torch.hstack(self.val_losses)
        val_losses = self.val_losses.cpu().detach().numpy()
        self.val_labels = torch.hstack(self.val_labels)
        val_labels = self.val_labels.cpu().detach().numpy()
        self.val_logits = torch.hstack(self.val_logits)
        val_logits = self.val_logits.cpu().detach().numpy()

        if not self.args.no_maha_threshold:
            val_df, mean, cov = compute_maha_threshold(val_labels, val_logits)

            self.maha_thresh = val_df["Threshold"][val_df["bal_acc"].argmax()]
            self.maha_mean = mean
            self.maha_cov = cov

        val_df, max_val_loss = compute_thresholds(val_losses, val_labels)
        self.max_val_loss = max_val_loss
        self.best_threshold = val_df["Threshold"][val_df["bal_acc"].argmax()]

        self.val_labels = []
        self.val_losses = []
        self.val_logits = []
        self.log("val_f1", val_df["F1"][val_df["bal_acc"].argmax()])
        self.log("threshold", self.best_threshold)
        self.log("max_score_threshold", self.max_val_loss)
        return self.log("loss/val", float(np.mean([x.cpu() for x in outputs])))

    def test_step(self, test_batch: Data, batch_idx):
        x, edge_index = self.process_batch(test_batch, batch_idx)

        y_hat = self.forward(x, edge_index)
        y = test_batch.y

        batchsize = len(test_batch.y)
        scores = torch.nn.MSELoss(reduction="none")(y_hat, x.view(-1)).view(batchsize, -1).mean(axis=1)

        self.test_scores.append(scores)
        self.test_labels.append(y)

        if not self.args.no_maha_threshold:
            maha_scores = np.array([mahalanobis(data, self.maha_mean, self.maha_cov) for data in y_hat.cpu().detach().numpy().reshape(test_batch.num_graphs, -1)])
            self.test_maha_scores.append(maha_scores)

    def test_epoch_end(self, outputs) -> None:
        compute_and_save_metrics(self)