# Mathematical Foundations: Rational Rate Conversion and Kaiser Windowing

---

## 1. Multirate Rational Sampling Rate Conversion

To convert a discrete-time signal $x[n]$ with sampling period $T_1$ to $y[m]$ with sampling period $T_2 = \frac{M}{L} T_1$:

### Step 1: Upsampling by $L$ (Zero-Insertion)
$$v[k] = \begin{cases} x[k / L] & \text{if } k \text{ is a multiple of } L \\ 0 & \text{otherwise} \end{cases}$$

In the frequency domain:

$$V(e^{j\omega}) = X(e^{j\omega L})$$

Zero-insertion compresses the spectrum by factor $L$, creating $L-1$ unwanted spectral images in the range $[0, 2\pi)$.

### Step 2: Ideal Low-Pass Filtering
An ideal digital low-pass filter $h[k]$ is applied to simultaneously remove the $L-1$ images and prevent aliasing before decimation:

$$w[k] = \sum_{l=-\infty}^\infty v[l] h[k - l]$$

The ideal filter gain is $L$, with cutoff frequency:

$$\omega_c = \min\left( \frac{\pi}{L}, \frac{\pi}{M} \right)$$

### Step 3: Downsampling by $M$ (Decimation)
$$y[m] = w[m M] = \sum_{l=-\infty}^\infty v[l] h[m M - l] = \sum_{n=-\infty}^\infty x[n] h[m M - n L]$$

---

## 2. Polyphase Decomposition Efficiency

Direct computation of $w[k]$ performs multiplications with zeros $L-1$ out of every $L$ times.
By decomposing the impulse response $h[k]$ into $L$ polyphase components:

$$e_p[n] = h[n L + p], \quad p \in \{0, 1, \dots, L-1\}$$

The output is computed by evaluating only non-zero terms:

$$y[m] = \sum_{n=-\infty}^\infty x[n] e_{(mM \bmod L)}\left[ \left\lfloor \frac{mM}{L} \right\rfloor - n \right]$$

This reduces computational complexity by a factor of $L$.

---

## 3. The Kaiser Window Parameterization

The Kaiser window provides closed-form control over sidelobe attenuation via parameter $\beta$:

$$w[n] = \frac{I_0\left( \beta \sqrt{1 - \left( \frac{2n}{N-1} - 1 \right)^2} \right)}{I_0(\beta)}, \quad 0 \le n \le N-1$$

Given a desired stopband attenuation $A = -20 \log_{10} \delta_s \text{ dB}$:

$$\beta = \begin{cases} 

0.1102 (A - 8.7) & \text{if } A > 50 \\
0.5842 (A - 21)^{0.4} + 0.07886 (A - 21) & \text{if } 21 \le A \le 50 \\
0.0 & \text{if } A < 21
\end{cases}

$$

The required filter order $N$ for normalized transition width $\Delta \omega$:

$$N \approx \frac{A - 8}{2.285 \Delta \omega}$$

For EduCasheer ($A = 80 \text{ dB}$, $\Delta \omega = 0.06\pi$), $\beta = 8.5555$ and $N = 16$ polyphase taps per branch.
