# Mathematical Foundations: Spectral Analysis and Mel Transformations

---

## 1. Orthogonality of the Discrete Fourier Basis

Let $\phi_k[n] = e^{j \frac{2\pi}{N} k n}$ for $k, n \in \{0, 1, \dots, N-1\}$.
The inner product of two basis vectors $\boldsymbol{\phi}_k$ and $\boldsymbol{\phi}_l$ is:

$$\langle \boldsymbol{\phi}_k, \boldsymbol{\phi}_l \rangle = \sum_{n=0}^{N-1} e^{j \frac{2\pi}{N} k n} \left( e^{j \frac{2\pi}{N} l n} \right)^* = \sum_{n=0}^{N-1} e^{j \frac{2\pi}{N} (k - l) n}$$

If $k = l$:

$$\langle \boldsymbol{\phi}_k, \boldsymbol{\phi}_k \rangle = \sum_{n=0}^{N-1} 1 = N$$

If $k \neq l$, using the finite geometric series sum with $r = e^{j \frac{2\pi}{N} (k-l)} \neq 1$:

$$\sum_{n=0}^{N-1} r^n = \frac{1 - r^N}{1 - r} = \frac{1 - e^{j 2\pi (k - l)}}{1 - e^{j \frac{2\pi}{N} (k - l)}} = \frac{1 - 1}{1 - e^{j \frac{2\pi}{N} (k - l)}} = 0$$

Thus, the basis vectors are strictly orthogonal:

$$\langle \boldsymbol{\phi}_k, \boldsymbol{\phi}_l \rangle = N \delta_{k, l}$$

---

## 2. Derivation of the Gabor Uncertainty Principle

Let continuous signal $x(t) \in L^2(\mathbb{R})$ have unit energy:

$$\int_{-\infty}^\infty |x(t)|^2 dt = 1$$

Define the mean time $\langle t \rangle$ and time variance $(\Delta t)^2$:

$$\langle t \rangle = \int_{-\infty}^\infty t |x(t)|^2 dt, \quad (\Delta t)^2 = \int_{-\infty}^\infty (t - \langle t \rangle)^2 |x(t)|^2 dt$$

By Parseval's theorem, the frequency variance $(\Delta \omega)^2$ is:

$$(\Delta \omega)^2 = \frac{1}{2\pi} \int_{-\infty}^\infty (\omega - \langle \omega \rangle)^2 |X(\omega)|^2 d\omega = \int_{-\infty}^\infty \left| \frac{d}{dt} \left( x(t) e^{-j \langle \omega \rangle t} \right) \right|^2 dt$$

Assuming without loss of generality that $\langle t \rangle = 0$ and $\langle \omega \rangle = 0$:

$$(\Delta t)^2 (\Delta \omega)^2 = \left( \int_{-\infty}^\infty t^2 |x(t)|^2 dt \right) \left( \int_{-\infty}^\infty \left| \frac{dx}{dt} \right|^2 dt \right)$$

Applying the Cauchy-Schwarz inequality:

$$(\Delta t)^2 (\Delta \omega)^2 \ge \left| \int_{-\infty}^\infty t x(t) \frac{dx}{dt} dt \right|^2$$

Using integration by parts on $\int_{-\infty}^\infty t \frac{d}{dt}(|x(t)|^2) dt$:

$$\int_{-\infty}^\infty t \cdot \frac{1}{2} \frac{d}{dt}(|x(t)|^2) dt = \left[ \frac{1}{2} t |x(t)|^2 \right]_{-\infty}^\infty - \frac{1}{2} \int_{-\infty}^\infty |x(t)|^2 dt = 0 - \frac{1}{2}(1) = -\frac{1}{2}$$

Taking the square yields:

$$(\Delta t)^2 (\Delta \omega)^2 \ge \left( -\frac{1}{2} \right)^2 = \frac{1}{4}$$

Converting from angular frequency $\omega = 2\pi f$ ($(\Delta \omega) = 2\pi \Delta f$):

$$\Delta t \cdot \Delta f \ge \frac{1}{4\pi} \quad \blacksquare$$

Equality holds if and only if $x(t)$ is a Gaussian pulse.
