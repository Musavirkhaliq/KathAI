# Mathematical Foundations: Non-Linear Least Squares Power-Law Fitting

---

## 1. Power-Law Estimation via Log-Transform

Let empirical data points be $(H_i, W_i)$ where $W_i = \text{WER}_i - \text{WER}_\infty$.
The power law model is:

$$W_i = A H_i^{-\alpha}$$

Taking the natural logarithm:

$$\ln W_i = \ln A - \alpha \ln H_i$$

Letting $y_i = \ln W_i, x_i = \ln H_i, c = \ln A$:

$$y_i = c - \alpha x_i$$

The optimal scaling exponent $\alpha$ is derived via closed-form linear regression:

$$\alpha = -\frac{N \sum x_i y_i - (\sum x_i)(\sum y_i)}{N \sum x_i^2 - (\sum x_i)^2}$$
$$c = \frac{\sum y_i + \alpha \sum x_i}{N}$$
$$A = e^c$$
