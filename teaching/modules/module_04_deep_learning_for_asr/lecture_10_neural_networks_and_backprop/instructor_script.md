# Instructor Delivery Script: Lecture 10 (Deep Neural Networks and Temporal Backpropagation)

## Timetable (90 Minutes)
```
00:00 - 00:20 | Deep Acoustic Modeling: Affine Maps and Non-Linearities (GELU vs. ReLU)
00:20 - 00:45 | Mathematical Proof: The Chain Rule Product and Gradient Singular Values
00:45 - 01:10 | Why BatchNorm Fails in Speech: The Padding Contamination Trap
01:10 - 01:25 | LayerNorm Mechanics and Numerical Stability
01:25 - 01:30 | Synthesis & Transition to Temporal Convolutions
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:45 | Gradient Flow on the Board
- Write the product on the Center Board:
  $$\prod_{l=1}^{24} \mathbf{W}^{(l) T}$$
- Look at the students:
  > "XLS-R has 24 Transformer layers. If each layer scales your gradient by just 1.1, then 1.1 to the power of 24 is 9.85.
  > If each layer scales your gradient by 0.9, then 0.9 to the power of 24 is 0.079.
  > Either your gradients explode into NaNs, or your bottom layers freeze and learn nothing.
  > How does modern deep learning train a 24-layer model on speech?
  > Two architectural miracles: **Residual Connections** ($\mathbf{x} + F(\mathbf{x})$) and **Layer Normalization**.
  > Residual connections create an unhindered gradient highway where $\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial F} + \mathbf{I}$, guaranteeing that the gradient can never decay to zero."
