# Mathematical Foundations: Acoustic Formants and F2 Transition Modeling

---

## 1. Acoustic Resonance in an Ideal Uniform Vocal Tract Tube

Modeling the human vocal tract as a lossless acoustic tube of length $L \approx 17.5 \text{ cm}$, open at the lips ($x = L$) and closed at the glottis ($x = 0$):

The pressure boundary conditions are:
$$\frac{\partial p}{\partial x}\Bigg|_{x=0} = 0 \quad (\text{zero particle velocity at glottis}), \quad p(L, t) = 0 \quad (\text{ambient pressure at lips})$$

The spatial pressure distribution satisfies:
$$p(x) = A \cos\left( \frac{2\pi f}{c} x \right)$$
Enforcing $p(L) = 0$ requires:
$$\cos\left( \frac{2\pi f}{c} L \right) = 0 \implies \frac{2\pi f}{c} L = \frac{(2n - 1)\pi}{2}, \quad n \in \{1, 2, 3, \dots\}$$

The resonant formant frequencies $F_n$ are:
$$F_n = \frac{(2n - 1) c}{4 L}$$
For $c = 34,300 \text{ cm/s}$ and $L = 17.5 \text{ cm}$:
- $F_1 = \frac{34300}{4 \times 17.5} = 490 \text{ Hz}$ (Neutral central schwa /ə/)
- $F_2 = 3 \times 490 = 1,470 \text{ Hz}$
- $F_3 = 5 \times 490 = 2,450 \text{ Hz}$

Notice that the neutral unconstricted tube corresponds precisely to the **Kashmiri central vowel /ə/** with $F_2 \approx 1470 \text{ Hz}$!

---

## 2. Perturbation Theory for Vowel Articulation

According to Chiba-Kajiyama-Stevens perturbation theory, localized constriction of cross-sectional area $\Delta A(x)$ alters formant frequencies according to the distribution of kinetic and potential acoustic energy:

$$\frac{\Delta F_n}{F_n} = \int_0^L \left( \frac{\Delta K_n(x)}{E_n} - \frac{\Delta U_n(x)}{E_n} \right) dx$$

- Constricting near a **volume velocity node** (pressure anti-node) increases acoustic stiffness, raising formant frequency.
- Constricting near a **volume velocity anti-node** (pressure node) increases acoustic inertia, lowering formant frequency.

For palatalization ($C^j$), constriction occurs at the hard palate ($x \approx 11 \text{ cm}$ from glottis), which coincides with a velocity anti-node for $F_2$, systematically driving $F_2$ upward toward $2,000 - 2,200 \text{ Hz}$.
