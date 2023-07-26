# ModeConv: A novel convolution for distinguishing anomalous and normal Structural Behavior



## Installation

Dependencies:

```
pip install torch --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

## How to use

### Training and evaluating a model
```
python train.py

optional arguments:
  -h, --help           show this help message and exit
  --model MODEL        options: 'ModeConvFast', 'ModeConvLaplace', 'ChebConv', 'AGCRN', 'MtGNN' (default: 'ModeConvFast')
  --dataset DATASET    options: 'simulated_smart_bridge', 'luxemburg' (default: 'luxemburg')
  --epochs N           (default: 50)
  --batch-size N       (default: 256)
  --lr lr              initial learning rate for optimizer e.g.: 1e-4 | 'auto' (default: 'auto')
  --num-layer N        (default: 3)
  --decoder DECODER    options: 'linear' for linear layer only decoder, 'custom': to use ModeConv/ChebConv/etc. layers in decoder (default: 'custom')
  --hidden-dim N       (default: 8)
  --bottleneck N       (default: 2)
  --no-maha-threshold  mahalanobis threshold calculation/evaluation is very slow; disabling saves 30min-2h in val and test on luxemburg dataset (default: True)
  --seed N             (default: 3407)
  --no-cuda            (default: False)

```

### Evaluating a model
```
python eval.py Path

positional arguments:
  Path                 path to pretrained weights, e.g. data/pretrained/luxemburg/ModeConvFast/GNN.statedict

optional arguments:
  -h, --help           show this help message and exit
  --model MODEL        options: 'ModeConvFast', 'ModeConvLaplace', 'ChebConv', 'AGCRN', 'MtGNN' (default: 'ModeConvFast')
  --dataset DATASET    options: 'simulated_smart_bridge', 'luxemburg' (default: 'luxemburg')
  --batch-size N       (default: 256)
  --num-layer N        (default: 3)
  --decoder DECODER    options: 'linear' for linear layer only decoder, 'custom': to use ModeConv/ChebConv/etc. layers in decoder (default: 'custom')
  --hidden-dim N       (default: 8)
  --bottleneck N       (default: 2)
  --no-maha-threshold  mahalanobis threshold calculation/evaluation is very slow; disabling saves 30min-2h in val and test on luxemburg dataset (default: True)
  --seed N             (default: 3407)
  --no-cuda            (default: False)

```

## Datasets

The csv files for the luxemburg dataset only contain about 5% of the full data used.
In the case of the Luxemburg dataset the 5% percent have even been to much to load it up in github. Therefore this link heads to the dataset: [https://drive.google.com/drive/folders/1KDAhKbqPGykxWSQzIfb0A4UDI5F5NZ9t?usp=drive_link)
Please store the Luxemburg dataset in the data folder of this repository.

The csv files for the simulated_smart_bridge dataset only contain about 6% of the full data.

## Results

### Luxemburg

```
python train.py --model {Model} --dataset luxemburg
```

| Model            | AUC   | F1    |
|----------------- |-------|-------|
| ModeConvFast     | 99.99 | 93.09 |
| ModeConvLaplace  | 92.16 | 73.74 |
| ChebConv         | 92.07 | 74.53 |
| AGCRN            | 98.18 | 86.67 |
| MtGNN            | 99.99 | 82.00 |

### Simulated Smart Bridge

```
python train.py --model {Model} --dataset simulated_smart_bridge
```

| Model            | AUC   | F1    |
|----------------- |-------|-------|
| ModeConvFast     | 92.23 | 87.93 |
| ModeConvLaplace  | 92.43 | 88.07 |
| ChebConv         | 82.15 | 83.89 |
| AGCRN            | 92.26 | 87.76 |
| MtGNN            | 91.19 | 86.78 |
