# ResNet Architecture Flow Diagrams

## 1. Overall ResNet Architecture Flow

```
Input Image (224x224x3)
        ↓
Initial Conv (7x7, 64)
        ↓
Max Pool (3x3)
        ↓
┌─────────────────────────────────────┐
│           Residual Blocks           │
│  ┌─────────┐  ┌─────────┐  ┌─────┐ │
│  │ Block 1 │→│ Block 2 │→│ ... │ │
│  └─────────┘  └─────────┘  └─────┘ │
└─────────────────────────────────────┘
        ↓
Global Average Pooling
        ↓
Fully Connected Layer
        ↓
Output (1000 classes)
```

## 2. Basic Residual Block Flow

```
Input
  │
  ├─────────────────┐
  │                 │
  ↓                 │
Conv1 (3x3)         │
  │                 │
  ↓                 │
BatchNorm           │
  │                 │
  ↓                 │
ReLU                │
  │                 │
  ↓                 │
Conv2 (3x3)         │
  │                 │
  ↓                 │
BatchNorm           │
  │                 │
  ↓                 │
  + ←───────────────┘
  │
  ↓
ReLU
  │
  ↓
Output
```

## 3. Bottleneck Residual Block Flow

```
Input
  │
  ├─────────────────────────────┐
  │                             │
  ↓                             │
Conv1 (1x1)                     │
  │                             │
  ↓                             │
BatchNorm                       │
  │                             │
  ↓                             │
ReLU                            │
  │                             │
  ↓                             │
Conv2 (3x3)                     │
  │                             │
  ↓                             │
BatchNorm                       │
  │                             │
  ↓                             │
ReLU                            │
  │                             │
  ↓                             │
Conv3 (1x1)                     │
  │                             │
  ↓                             │
BatchNorm                       │
  │                             │
  ↓                             │
  + ←───────────────────────────┘
  │
  ↓
ReLU
  │
  ↓
Output
```

## 4. ResNet-50 Architecture Flow

```
Input (224x224x3)
        ↓
Conv1 (7x7, 64)
        ↓
Max Pool (3x3)
        ↓
┌─────────────────────────────────────┐
│           Stage 1 (3 blocks)        │
│  ┌─────────┐  ┌─────────┐  ┌─────┐ │
│  │Bottleneck│→│Bottleneck│→│Bottle│ │
│  └─────────┘  └─────────┘  └─────┘ │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│           Stage 2 (4 blocks)        │
│  ┌─────────┐  ┌─────────┐  ┌─────┐ │
│  │Bottleneck│→│Bottleneck│→│ ... │ │
│  └─────────┘  └─────────┘  └─────┘ │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│           Stage 3 (6 blocks)        │
│  ┌─────────┐  ┌─────────┐  ┌─────┐ │
│  │Bottleneck│→│Bottleneck│→│ ... │ │
│  └─────────┘  └─────────┘  └─────┘ │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│           Stage 4 (3 blocks)        │
│  ┌─────────┐  ┌─────────┐  ┌─────┐ │
│  │Bottleneck│→│Bottleneck│→│Bottle│ │
│  └─────────┘  └─────────┘  └─────┘ │
└─────────────────────────────────────┘
        ↓
Global Average Pooling
        ↓
Fully Connected Layer
        ↓
Output (1000 classes)
```

## 5. Feature Map Dimensions Flow

```
Input: 224x224x3
        ↓
Conv1: 112x112x64
        ↓
Stage 1: 56x56x256
        ↓
Stage 2: 28x28x512
        ↓
Stage 3: 14x14x1024
        ↓
Stage 4: 7x7x2048
        ↓
Global Pool: 1x1x2048
        ↓
FC Layer: 1000
```

## 6. Skip Connection Types

```
1. Identity Skip Connection
Input → Conv → BN → ReLU → Conv → BN → Add → ReLU → Output
   ↓                                    ↑
   └────────────────────────────────────┘

2. Projection Skip Connection
Input → Conv → BN → ReLU → Conv → BN → Add → ReLU → Output
   ↓                                    ↑
   └────→ Conv → BN ───────────────────┘
```

## 7. Training Flow with Skip Connections

```
Forward Pass:
Input → Main Path → Add → Output
   ↓              ↑
   └──────────────┘

Backward Pass:
Gradient → Main Path → Add → Input Gradient
   ↓                ↑
   └────────────────┘
```

These diagrams illustrate the key components and data flow in ResNet architecture, showing how skip connections enable effective training of deep networks. 