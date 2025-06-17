# CNN Architecture Diagrams

## 1. Basic CNN Architecture

```
Input Image (32x32x3) → Conv Layer → Pooling → Conv Layer → Pooling → FC Layer → Output
```

## 2. Convolution Operation

```
Input Feature Map (5x5)    Kernel (3x3)    Output Feature Map (3x3)
+---+---+---+---+---+     +---+---+---+   +---+---+---+
| 1 | 2 | 3 | 4 | 5 |     | 1 | 0 | 1 |   | 12| 15| 18|
+---+---+---+---+---+     +---+---+---+   +---+---+---+
| 2 | 3 | 4 | 5 | 6 |     | 0 | 1 | 0 |   | 15| 18| 21|
+---+---+---+---+---+     +---+---+---+   +---+---+---+
| 3 | 4 | 5 | 6 | 7 |     | 1 | 0 | 1 |   | 18| 21| 24|
+---+---+---+---+---+     +---+---+---+   +---+---+---+
| 4 | 5 | 6 | 7 | 8 |
+---+---+---+---+---+
| 5 | 6 | 7 | 8 | 9 |
+---+---+---+---+---+
```

## 3. Max Pooling Operation

```
Input (4x4)           Output (2x2)
+---+---+---+---+    +---+---+
| 1 | 2 | 3 | 4 |    | 6 | 8 |
+---+---+---+---+    +---+---+
| 5 | 6 | 7 | 8 |    |14 |16 |
+---+---+---+---+    +---+---+
| 9 |10 |11 |12 |
+---+---+---+---+
|13 |14 |15 |16 |
+---+---+---+---+
```

## 4. LeNet-5 Architecture

```
Input → Conv1 → Pool1 → Conv2 → Pool2 → FC1 → FC2 → Output
(32x32)  (28x28)  (14x14)  (10x10)  (5x5)   (120)  (84)   (10)
```

## 5. Feature Map Visualization

```
Layer 1 (Edges)    Layer 2 (Shapes)    Layer 3 (Objects)
+---+---+---+     +---+---+---+      +---+---+---+
|   |   |   |     |   |   |   |      |   |   |   |
+---+---+---+     +---+---+---+      +---+---+---+
|   |   |   |     |   |   |   |      |   |   |   |
+---+---+---+     +---+---+---+      +---+---+---+
|   |   |   |     |   |   |   |      |   |   |   |
+---+---+---+     +---+---+---+      +---+---+---+
```

## 6. ResNet Skip Connection

```
Input → Conv1 → Conv2 → Conv3 → Output
   ↓                    ↑
   └────────────────────┘
      Skip Connection
```

## 7. CNN Training Process

```
Input Data → Forward Pass → Loss Calculation → Backpropagation → Weight Update
   ↑                                                              ↓
   └──────────────────────────────────────────────────────────────┘
```

## 8. Feature Hierarchy

```
Low-level Features → Mid-level Features → High-level Features
(Edges, Colors)     (Shapes, Patterns)   (Objects, Scenes)
```

These diagrams provide a visual representation of key CNN concepts and architectures. They complement the detailed explanations in the main white paper. 