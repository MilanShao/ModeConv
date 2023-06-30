import json

import networkx
import numpy as np
import pandas as pd
import torch
import tqdm
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx


#create sample
def create_sample(
        strain,
        temp,
        inclination,
        cov,
        labels,
        edge_index,
):

    sample = Data(x=torch.concatenate((inclination, temp), dim=-1), edge_index=edge_index, y=labels)
    sample.strain = strain
    sample.cov = cov
    return sample


def create_sample_luxem(
        acceleration,
        temp,
        voltage,
        disp,
        cov,
        labels,
        edge_index
):
    sample = Data(x=torch.concatenate((voltage, temp), dim=-1), edge_index=edge_index, y=labels)
    sample.acc = acceleration
    sample.disp = disp
    sample.cov = cov
    return sample


def batch_cov(points):
    """https://stackoverflow.com/a/71357620"""
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)


class SimulatedSmartBridgeDataset(InMemoryDataset):
    def __init__(self, root, mode="train", data="short", transform=None, pre_transform=None, pre_filter=None):
        self.mode = mode
        self.data = data
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.mode == "train":
            return ["train_data.pt"]
        elif self.mode == "val":
            return ["val_data.pt"]
        elif self.mode == "test":
            return ["test_data.pt"]

    def process(self):
        data = pd.read_csv(f"data/SimulatedSmartBridge/simulated_smart_bridge_{self.mode}.csv")
        inclination = data.values[:, :3]
        temp = data.values[:, 3:4]
        strain = data.values[:, 4:16]
        labels = data.values[:, -1]

        num_sensors = 12
        G = networkx.complete_graph(num_sensors)
        pyg_graph: Data = from_networkx(G)

        lookback = 5
        tau = 5
        num = lookback * tau

        # write metadata
        if self.mode == "train":
            stats = {}
            stats["inclination"] = {"mean": inclination.mean(), "std": inclination.std(), "min": inclination.min(), "max": inclination.max()}
            stats["temp"] = {"mean": temp.mean(), "std": temp.std(), "min": temp.min(), "max": temp.max()}
            stats["strain"] = {"mean": strain.mean(), "std": strain.std(), "min": strain.min(), "max": strain.max()}
            with open(f'{self.processed_dir}/metadata.json', 'w') as fp:
                json.dump(stats, fp, indent=4)

        strain = torch.Tensor(strain.reshape((-1, lookback, tau, num_sensors)))
        cov = torch.zeros(
            (strain.shape[0] * strain.shape[1], strain.shape[-1], strain.shape[-1]))
        strain2 = torch.nn.functional.pad(strain.view((-1, strain.shape[-2], strain.shape[-1])),
                                                (0, 0, 0, 0, 0, 1))
        for i in range(num_sensors):
            strain3 = strain2.clone()
            strain3[0:-1, :, np.delete(np.arange(num_sensors), i)] = strain3[1:, :,
                                                                           np.delete(np.arange(num_sensors), i)]
            cov[:, i, :] = batch_cov(strain3)[:-1, i]
        for i in range(num_sensors):
            s = strain2[:-1, :, i].unsqueeze(-1)
            s_lag = strain2[1:, :, i].unsqueeze(-1)
            s = torch.concatenate((s, s_lag), -1)
            test = batch_cov(s)
            cov[:, i, i] = test[:, 0, 1]

        cov = cov.view((strain.shape[0], strain.shape[1], strain.shape[-1], strain.shape[-1]))

        cov = (cov - (-370000)) / (370000 - (-370000))

        strain = strain.reshape((-1, num, num_sensors))
        temp = torch.Tensor(temp.reshape((-1, 1, num, 1))).repeat(1, num_sensors, 1, 1).squeeze(-1)
        inclination = torch.Tensor(np.repeat(inclination.reshape((-1, num, 3)), [4,4,4], axis=2).transpose(0,2,1))
        labels = labels[num - 1::num]
        data_list = []
        for i in tqdm.tqdm(range(len(labels))):
            sample = create_sample(
                strain[i],
                temp[i],
                inclination[i],
                cov[i],
                labels[i],
                pyg_graph.edge_index,
            )

            # Read data into huge `Data` list.
            data_list.append(sample)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class LuxemburgDataset(InMemoryDataset):
    def __init__(self, root, mode="train", transform=None, pre_transform=None, pre_filter=None):
        self.mode = mode
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.mode == "train":
            return ["train_data.pt"]
        elif self.mode == "val":
            return ["val_data.pt"]
        elif self.mode == "test":
            return ["test_data.pt"]

    def process(self):
        data = pd.read_csv(f"data/luxemburg/lux_{self.mode}.csv")
        voltage = data.values[:, :4]
        temp = data.values[:, 4:5]
        disp = data.values[:, 5:13]
        acceleration = data.values[:, 13:39]
        labels = data.values[:, -1]

        num_sensors = 26
        G = networkx.complete_graph(num_sensors)
        pyg_graph: Data = from_networkx(G)

        lookback = 5
        tau = 5
        num = lookback * tau

        # write metadata
        if self.mode == "train":
            stats = {}
            stats["voltage"] = {"mean": list(voltage.mean(0)), "std": list(voltage.std(0)), "min": list(voltage.min(0)), "max": list(voltage.max(0))}
            stats["temp"] = {"mean": list(temp.mean(0)), "std": list(temp.std(0)), "min": list(temp.min(0)), "max": list(temp.max(0))}
            stats["disp"] = {"mean": list(disp.mean(0)), "std": list(disp.std(0)), "min": list(disp.min(0)), "max": list(disp.max(0))}
            stats["acceleration"] = {"mean": acceleration.mean(), "std": acceleration.std(), "min": acceleration.min(), "max": acceleration.max()}
            with open(f'{self.processed_dir}/metadata.json', 'w') as fp:
                json.dump(stats, fp, indent=4)

        acceleration = torch.Tensor(acceleration.reshape((-1, lookback, tau, num_sensors)))
        cov = torch.zeros(
            (acceleration.shape[0] * acceleration.shape[1], acceleration.shape[-1], acceleration.shape[-1]))
        acceleration2 = torch.nn.functional.pad(acceleration.view((-1, acceleration.shape[-2], acceleration.shape[-1])),
                                                (0, 0, 0, 0, 0, 1))
        for i in range(num_sensors):
            acceleration3 = acceleration2.clone()
            acceleration3[0:-1, :, np.delete(np.arange(num_sensors), i)] = acceleration3[1:, :,
                                                                           np.delete(np.arange(num_sensors), i)]
            cov[:, i, :] = batch_cov(acceleration3)[:-1, i]
        for i in range(num_sensors):
            acc = acceleration2[:-1, :, i].unsqueeze(-1)
            acc_lag = acceleration2[1:, :, i].unsqueeze(-1)
            acc = torch.concatenate((acc, acc_lag), -1)
            test = batch_cov(acc)
            cov[:, i, i] = test[:, 0, 1]

        cov = cov.view((acceleration.shape[0], acceleration.shape[1], acceleration.shape[-1], acceleration.shape[-1]))

        cov = (cov - (-0.001)) / (0.001 - (-0.001))

        acceleration = acceleration.reshape((-1, num, num_sensors))
        temp = torch.Tensor(temp.reshape((-1, num, 1)))
        voltage = torch.Tensor(voltage.reshape((-1, num, 4)))
        disp = torch.Tensor(disp.reshape((-1, num, 8)))
        labels = labels[num-1::num]
        data_list = []
        for i in tqdm.tqdm(range(len(acceleration))):
            sample = create_sample_luxem(
                acceleration[i],
                temp[i],
                voltage[i],
                disp[i],
                cov[i],
                labels[i],
                pyg_graph.edge_index,
            )

            # Read data into huge `Data` list.
            data_list.append(sample)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
