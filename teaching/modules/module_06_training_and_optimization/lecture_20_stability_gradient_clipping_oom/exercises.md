# Exercises: Lecture 20 (Numerical Stability and OOM Recovery)

---

## Exercise 1: Memory Scaling Analysis
For $B = 8, H = 16, L = 24$, calculate the memory required for storing attention score matrices when:
1. $T_{\max} = 250 \text{ frames}$ (5.0 seconds).
2. $T_{\max} = 1,000 \text{ frames}$ (20.0 seconds).
3. What is the ratio of attention memory between the 20s clip and the 5s clip?

---

## Exercise 2: NaN Contamination Dynamics
Explain step-by-step how a single NaN gradient in step $t$ corrupts the entire training run if the optimizer updates the weights without NaN detection.
