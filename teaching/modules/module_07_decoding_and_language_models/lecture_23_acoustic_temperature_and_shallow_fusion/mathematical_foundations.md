# Mathematical Foundations: Temperature Scaling and Brevity Bias Derivations

---

## 1. Temperature Calibration Optimization

Let validation set contain $N$ frames with ground-truth alignment labels $y_t^*$.
Optimal temperature $T^*$ minimizes negative log-likelihood (NLL):

$$T^* = \arg\min_T -\sum_{t=1}^N \log \left( \frac{\exp(z_{y_t^*}^t / T)}{\sum_k \exp(z_k^t / T)} \right)$$

Solved via 1D line search (Brent's method) or gradient descent over scalar $T$.

---

## 2. Derivation of the Brevity Bias

In character-level decoding, a word of length $L$ characters requires $L$ emission transitions.
Because each transition probability $P(c) < 1.0$, log-probabilities are negative ($\log P < 0$).
Consequently, longer words accumulate larger negative penalties:

$$\log P_{\text{AM}}(\text{long word}) \ll \log P_{\text{AM}}(\text{short word})$$

Without compensation, the beam search systematically favors deleting words or choosing shorter erroneous synonyms.
Adding $\beta |\mathbf{w}|$ adds a positive reward for each completed word, mathematically counteracting acoustic deletion bias.
