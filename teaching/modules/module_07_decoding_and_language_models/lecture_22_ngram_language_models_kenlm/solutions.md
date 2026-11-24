# Solutions: Lecture 22 (N-Gram Language Models and KenLM)

---

## Solution to Exercise 1
1. Continuation probability:
   $$P_{\text{continuation}}(\text{"فرانسسکو"}) = \frac{1}{20,000} = \mathbf{0.00005 \text{ (5 \times 10^{-5})}}$$
2. Unigram MLE probability:
   $$P_{\text{MLE}}(\text{"فرانسسکو"}) = \frac{500}{100,000} = \mathbf{0.0050 \text{ (5 \times 10^{-3})}}$$
3. Notice that $P_{\text{MLE}}$ is **100 times higher** than $P_{\text{continuation}}$! In an unseen context (e.g., after the Kashmiri word for "mountain"), standard MLE would assign a dangerously high backoff probability to "Francisco" simply because it was frequent, whereas Kneser-Ney penalizes it because it lacks context diversity.

---

## Solution to Exercise 2
1. Cross-entropy:
   $$H = -\frac{1}{500} \sum_{i=1}^{500} \log_{10} P(w_i) = -\frac{-1,250.0}{500} = \mathbf{2.50}$$
2. Perplexity:
   $$\text{PPL} = 10^H = 10^{2.50} = \mathbf{316.23}$$
