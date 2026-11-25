# Mathematical Foundations: Sampling Theory and Quantization

---

## 1. Fourier Transform of the Periodic Dirac Impulse Train

Let the periodic impulse train with period $T_s$ be:

$$p(t) = \sum_{n=-\infty}^\infty \delta(t - n T_s)$$

Since $p(t)$ is periodic with fundamental period $T_s$, it can be expanded as a complex Fourier series:

$$p(t) = \sum_{k=-\infty}^\infty c_k e^{j 2\pi k f_s t}, \quad f_s = \frac{1}{T_s}$$

The Fourier series coefficients $c_k$ are:

$$c_k = \frac{1}{T_s} \int_{-T_s/2}^{T_s/2} p(t) e^{-j 2\pi k f_s t} dt = \frac{1}{T_s} \int_{-T_s/2}^{T_s/2} \delta(t) e^{-j 2\pi k f_s t} dt = \frac{1}{T_s}$$

Taking the Continuous-Time Fourier Transform of both sides:

$$\mathcal{F}\{p(t)\} = \sum_{k=-\infty}^\infty \frac{1}{T_s} \mathcal{F}\left\{ e^{j 2\pi k f_s t} \right\} = \frac{1}{T_s} \sum_{k=-\infty}^\infty \delta(f - k f_s)$$

---

## 2. Derivation of the Sampling Spectrum

Let $x_s(t) = x(t) p(t)$. By the convolution theorem:

$$\mathcal{F}\{x_s(t)\} = X(f) * P(f) = X(f) * \left[ \frac{1}{T_s} \sum_{k=-\infty}^\infty \delta(f - k f_s) \right]$$

Applying the sifting property of the Dirac delta function:

$$X_s(f) = \frac{1}{T_s} \sum_{k=-\infty}^\infty \int_{-\infty}^\infty X(\nu) \delta(f - k f_s - \nu) d\nu = \frac{1}{T_s} \sum_{k=-\infty}^\infty X(f - k f_s)$$

---

## 3. The Ideal Reconstruction (Whittaker-Shannon Interpolation)

If $X(f) = 0$ for $|f| \ge f_s/2$, the original spectrum $X(f)$ can be isolated by multiplying $X_s(f)$ with an ideal low-pass brickwall filter:

$$H_{\text{ideal}}(f) = \begin{cases} T_s & \text{if } |f| < \frac{f_s}{2} \\ 0 & \text{otherwise} \end{cases}$$

In the time domain, the impulse response is the sinc function:

$$h_{\text{ideal}}(t) = \mathcal{F}^{-1}\{H_{\text{ideal}}(f)\} = \int_{-f_s/2}^{f_s/2} T_s e^{j 2\pi f t} df = \frac{\sin(\pi f_s t)}{\pi f_s t} = \text{sinc}(f_s t)$$

Convolving the discrete samples with the sinc kernel reconstructs the continuous signal perfectly:

$$x(t) = x_s(t) * h_{\text{ideal}}(t) = \sum_{n=-\infty}^\infty x(n T_s) \text{sinc}\left( \frac{t - n T_s}{T_s} \right)$$
