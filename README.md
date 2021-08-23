# Fork of SimCLRv2-Pytorch

This fork slightly modifies the code to support ResNet18 models from the official SimCLRv2 repo.
The implementation was tested:
- CIFAR10 & CIFAR100 SimCLR models were trained using the code & commands provided in the Google's `simclr` repo as of [this commit](https://github.com/google-research/simclr/tree/dec99a81a4ceccb0a5a893afecbc2ee18f1d76c3);
- Only unsupervised pretraining phase was executed; no linear head fine-tuning was performed;
- The last checkpoint was converted to PyTorch code using the code in this repository (see `convert_resnet18.ipynb`);
- The converted (frozen) models were loaded into PyTorch and with attached linear classification head I was able to achieve 95% and 74.4% validation accuracy on CIFAR10 and CIFAR100 respectively.

# Original ReadMe content (below)

Pretrained SimCLRv2 models in Pytorch.

```python
python download.py r152_3x_sk1
python convert.py r152_3x_sk1/model.ckpt-250228 [--ema]
python verify.py r152_3x_sk1.pth
```

| Model | Pytorch | TF |
| :-------------: |:-------------:| :-----:|
| r50_1x_sk0 | 70.97 | 71.7 |
| r50_1x_sk1 | 73.79 | 74.6 |
| r152_3x_sk1 | 79.12 | 79.8 |
