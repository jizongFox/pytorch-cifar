# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites

- Python 3.9+
- PyTorch 2.x, a matching torchvision build, and NumPy
- CUDA and NCCL for distributed training

## Training

```bash
python main.py
```

`main.py` uses CUDA when available and otherwise runs on CPU. For one distributed
process per visible GPU, use `main_dist.py` with the same training arguments.

```bash
CUDA_VISIBLE_DEVICES=0,1 python main_dist.py
```

## Vision Transformer comparison

Both classifiers use 4×4 patches, a class token, learned position embeddings,
and width-128 pre-norm encoder blocks with four attention heads. `--depth`
sets the number of block applications. The default comparison isolates weight
sharing at fixed depth. [Déjà View, §4.2](https://arxiv.org/html/2605.30215v2#S4.SS2)

| Model | Encoder blocks stored | Block applications | Trainable parameters |
| ----- | --------------------: | -----------------: | -------------------: |
| `vit` | 6 | 6 | 1,205,898 |
| `looped-vit` | 1 | 6 | 214,538 |

At depth six the models execute the same attention and MLP operations.
Reusing the block reduces parameter storage.

Both commands train from scratch for 200 epochs with the same CIFAR-10
augmentations, AdamW at `3e-4`, and cosine learning-rate decay. They print
trainable parameter counts and test top-1 accuracy.

```bash
python main.py --model vit --depth 6 --seed 0
python main.py --model looped-vit --depth 6 --seed 0 --eval-loop-steps 1 2 4
```

The loop-step sweep evaluates shorter exits from the same fixed-depth trained
model after its final epoch. A separate `vit` run with `--depth 1` gives a
parameter-matched comparison with the six-step looped model.

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [RegNetX_200MF](https://arxiv.org/abs/2003.13678)     | 94.24%      |
| [RegNetY_400MF](https://arxiv.org/abs/2003.13678)     | 94.29%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [SimpleDLA](https://arxiv.org/abs/1707.064)           | 94.89%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |
| [DLA](https://arxiv.org/pdf/1707.06484.pdf)           | 95.47%      |

