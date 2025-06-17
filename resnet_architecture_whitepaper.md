# ResNet Architecture: A Deep Dive into Residual Networks

## Table of Contents
1. [Introduction](#introduction)
2. [The Vanishing Gradient Problem](#vanishing-gradient)
3. [ResNet Architecture Overview](#architecture-overview)
4. [Residual Blocks](#residual-blocks)
5. [Different ResNet Variants](#resnet-variants)
6. [Implementation Details](#implementation)
7. [Training and Optimization](#training)
8. [Applications and Impact](#applications)
9. [Conclusion](#conclusion)

## Introduction

ResNet (Residual Neural Network) is a groundbreaking deep learning architecture introduced by Microsoft Research in 2015. It revolutionized deep learning by enabling the training of extremely deep networks (up to 152 layers) through the introduction of skip connections, solving the vanishing gradient problem that had previously limited network depth.

## The Vanishing Gradient Problem

### Traditional Deep Networks
```
Input → Layer1 → Layer2 → Layer3 → ... → LayerN → Output
```

### Issues with Deep Networks
- Gradients become extremely small during backpropagation
- Network performance degrades with increasing depth
- Harder to optimize deeper networks

## ResNet Architecture Overview

### Basic Structure
```
Input → Initial Conv → Residual Blocks → Global Avg Pool → FC Layer → Output
```

### Key Components
1. Initial Convolution Layer
2. Residual Blocks
3. Global Average Pooling
4. Fully Connected Layer

## Residual Blocks

### Basic Residual Block
```
Input → Conv1 → BN → ReLU → Conv2 → BN → Add → ReLU → Output
   ↓                                    ↑
   └────────────────────────────────────┘
```

### Bottleneck Residual Block
```
Input → 1x1 Conv → BN → ReLU → 3x3 Conv → BN → ReLU → 1x1 Conv → BN → Add → ReLU → Output
   ↓                                                                        ↑
   └────────────────────────────────────────────────────────────────────────┘
```

## Different ResNet Variants

### ResNet-18/34
```
Basic Block Structure:
Input → [Conv3x3, 64] → [Basic Block × 2] → [Basic Block × 2] → [Basic Block × 2] → [Basic Block × 2] → Output
```

### ResNet-50/101/152
```
Bottleneck Block Structure:
Input → [Conv1x1, 64] → [Bottleneck Block × 3] → [Bottleneck Block × 4] → [Bottleneck Block × 6] → [Bottleneck Block × 3] → Output
```

## Implementation Details

### Basic Residual Block Implementation
```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
```

### Complete ResNet Implementation
```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```

## Training and Optimization

### Training Process
```
Data → Forward Pass → Loss Calculation → Backpropagation → Weight Update
   ↑                                                      ↓
   └──────────────────────────────────────────────────────┘
```

### Optimization Techniques
1. Batch Normalization
2. Weight Decay
3. Learning Rate Scheduling
4. Momentum

## Applications and Impact

### Key Applications
1. Image Classification
2. Object Detection
3. Image Segmentation
4. Feature Extraction

### Impact on Deep Learning
- Enabled training of much deeper networks
- Improved performance on various tasks
- Inspired new architectures
- Became backbone for many computer vision systems

## Conclusion

ResNet's introduction of skip connections has fundamentally changed how we think about deep neural networks. Its ability to train extremely deep networks effectively has made it one of the most influential architectures in deep learning history.

---

## References

1. He, K., et al. (2016). Deep Residual Learning for Image Recognition.
2. He, K., et al. (2016). Identity Mappings in Deep Residual Networks.
3. He, K., et al. (2017). Mask R-CNN. 