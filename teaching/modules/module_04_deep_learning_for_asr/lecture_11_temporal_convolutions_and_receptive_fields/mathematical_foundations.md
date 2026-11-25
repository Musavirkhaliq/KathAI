# Mathematical Foundations: Receptive Field and Feature Length Slicing

---

## 1. General Recursive Receptive Field Formula

Let a cascade of $L$ 1D convolutional layers have kernel sizes $K_l$, strides $S_l$, and dilations $D_l$.
The effective kernel size $K_l'$ considering dilation is:

$$K_l' = K_l + (K_l - 1)(D_l - 1) = 1 + D_l(K_l - 1)$$

The cumulative stride $J_l$ after layer $l$ is:

$$J_l = \prod_{i=1}^l S_i, \quad J_0 = 1$$

The total receptive field $R_l$ satisfies the difference equation:

$$R_l = R_{l-1} + (K_l' - 1) J_{l-1}, \quad R_0 = 1$$

Expanding the telescoping sum:

$$R_L = 1 + \sum_{l=1}^L (K_l' - 1) \prod_{i=1}^{l-1} S_i$$

---

## 2. Output Sequence Length Equation

Given input length $T_\text{in}$, the output sequence length after $L$ convolutional layers with zero padding is:

$$T_{l} = \left\lfloor \frac{T_{l-1} - K_l}{S_l} \right\rfloor + 1$$
