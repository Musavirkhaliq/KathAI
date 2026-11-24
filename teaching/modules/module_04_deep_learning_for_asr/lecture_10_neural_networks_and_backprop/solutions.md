# Solutions: Lecture 10 (Deep Neural Networks and Backpropagation)

---

## Solution to Exercise 1
1. Definition: $\text{GELU}(x) = x \Phi(x)$, where $\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^x e^{-t^2/2} dt$.
2. Derivative:
   $$\frac{d}{dx}\text{GELU}(x) = \Phi(x) + x \Phi'(x) = \Phi(x) + x \frac{1}{\sqrt{2\pi}} e^{-x^2 / 2}$$
3. Values:
   - At $x = 0$: $\Phi(0) = 0.5$, $\Phi'(0) = \frac{1}{\sqrt{2\pi}}$, so $\frac{d}{dx}\text{GELU}(0) = 0.5 + 0 = \mathbf{0.5}$.
   - At $x = -2.0$: $\Phi(-2.0) \approx 0.0228$, $\Phi'(-2.0) = \frac{1}{\sqrt{2\pi}} e^{-2} \approx 0.0540$.
     $$\frac{d}{dx}\text{GELU}(-2.0) = 0.0228 + (-2.0)(0.0540) = 0.0228 - 0.108 = \mathbf{-0.0852}$$
   Unlike ReLU which outputs strictly 0 gradient for all negative inputs (causing dead neurons), GELU has non-zero gradients in the negative region, allowing recovery from suboptimal initialization.

---

## Solution to Exercise 2
1. For $\mathbf{y} = \alpha \mathbf{x}$ ($\alpha > 0$):
   $$\mu_y = \alpha \mu_x, \quad \sigma_y^2 = \alpha^2 \sigma_x^2 \implies \sigma_y = \alpha \sigma_x$$
   $$\hat{y}_i = \frac{\alpha x_i - \alpha \mu_x}{\alpha \sigma_x} = \frac{x_i - \mu_x}{\sigma_x} = \hat{x}_i \quad \blacksquare$$
2. For $\mathbf{y} = \mathbf{x} + c \mathbf{1}$:
   $$\mu_y = \mu_x + c, \quad \sigma_y^2 = \frac{1}{D} \sum (x_i + c - (\mu_x + c))^2 = \sigma_x^2$$
   $$\hat{y}_i = \frac{(x_i + c) - (\mu_x + c)}{\sigma_x} = \frac{x_i - \mu_x}{\sigma_x} = \hat{x}_i \quad \blacksquare$$
