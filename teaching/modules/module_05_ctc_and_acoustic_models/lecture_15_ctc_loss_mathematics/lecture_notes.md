# Lecture 15: Connectionist Temporal Classification (CTC) Mathematics

---

## 1. The Extended Target Sequence and Alignment Lattice

Let $\mathbf{l} = (l_1, l_2, \dots, l_U)$ be the target label sequence of length $U$ over alphabet $\Omega$.
CTC introduces the non-emitting blank symbol $\epsilon$ ($\Omega' = \Omega \cup \{\epsilon\}$).

To account for blanks occurring before, between, and after every label, CTC defines the **extended target sequence** $\mathbf{l}'$ of length $U' = 2U + 1$:

$$\mathbf{l}' = (\epsilon, l_1, \epsilon, l_2, \epsilon, \dots, \epsilon, l_U, \epsilon)$$

### Index Mapping:
- Odd indices ($s = 1, 3, 5, \dots, 2U+1$): Always correspond to blank $\epsilon$.
- Even indices ($s = 2, 4, 6, \dots, 2U$): Correspond to actual target characters $l_{s/2}$.

---

## 2. The Forward Variable $\alpha_t(s)$ Recursion

The forward variable $\alpha_t(s)$ represents the total probability mass of all valid partial alignments of length $t$ that map onto the prefix sub-sequence $\mathbf{l}'_{1:s}$:

$$\alpha_t(s) = \sum_{\substack{\boldsymbol{\pi}_{1:t} \in (\Omega')^t \\ \mathcal{B}(\boldsymbol{\pi}_{1:t}) = \mathbf{l}'_{1:s}}} \prod_{\tau=1}^t y_{\pi_\tau}^\tau$$

### Base Case ($t = 1$):
At the first time frame, an alignment can only start on the first blank or the first label:
$$\alpha_1(1) = y_\epsilon^1$$
$$\alpha_1(2) = y_{l_1}^1$$
$$\alpha_1(s) = 0 \quad \forall s > 2$$

### Recursive Step ($t > 1$):
At time step $t$, the transition to state $s$ depends on whether state $s$ is a blank, a duplicate label, or a new distinct label:

**Case 1: State $s$ is blank ($\mathbf{l}'_s = \epsilon$) or identical to the previous label ($\mathbf{l}'_s = \mathbf{l}'_{s-2}$)**:
The alignment could have stayed at state $s$ or transitioned from state $s-1$:
$$\alpha_t(s) = \left( \alpha_{t-1}(s) + \alpha_{t-1}(s-1) \right) y_{\mathbf{l}'_s}^t$$

**Case 2: State $s$ is a distinct label ($\mathbf{l}'_s \neq \epsilon$ and $\mathbf{l}'_s \neq \mathbf{l}'_{s-2}$)**:
The alignment could have stayed at state $s$, moved from the immediately preceding blank $s-1$, or **skipped the blank entirely** from state $s-2$:
$$\alpha_t(s) = \left( \alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \alpha_{t-1}(s-2) \right) y_{\mathbf{l}'_s}^t$$

---

## 3. Total Likelihood and Loss Definition

At the final frame $T$, a valid alignment must end either on the final label ($s = 2U$) or the final blank ($s = 2U + 1$):

$$P(\mathbf{l} \mid \mathbf{X}) = \alpha_T(2U) + \alpha_T(2U + 1)$$

The **CTC Loss** is defined as the negative log-likelihood of the ground-truth sequence:
$$\mathcal{L}_{\text{CTC}} = -\log P(\mathbf{l} \mid \mathbf{X}) = -\log \left( \alpha_T(2U) + \alpha_T(2U + 1) \right)$$

---

## 4. Analytical Logit Gradients: Why CTC Produces Peaky Distributions

Let $z_k^t$ be the unnormalized logit for character $k$ at time $t$, where $y_k^t = \text{Softmax}(z_k^t)$.
By the chain rule, the gradient of $\mathcal{L}_{\text{CTC}}$ with respect to logit $z_k^t$ is:

$$\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial z_k^t} = y_k^t - \frac{1}{P(\mathbf{l} \mid \mathbf{X})} \sum_{s \in \text{lab}(\mathbf{l}', k)} \alpha_t(s) \beta_t(s)$$

where $\text{lab}(\mathbf{l}', k) = \{s \mid \mathbf{l}'_s = k\}$, and $\beta_t(s)$ is the backward CTC variable.

### The Peaky Behavior Phenomenon:
Notice the elegant structure of the gradient:
$$\frac{\partial \mathcal{L}_{\text{CTC}}}{\partial z_k^t} = y_k^t - \gamma_t(k)$$
where $\gamma_t(k)$ is the posterior probability that symbol $k$ was emitted at frame $t$.
- When the acoustic evidence is ambiguous, the model prefers emitting the blank token $\epsilon$ because the blank is valid across multiple paths in the lattice.
- When an acoustic transition is sharp, the model concentrates all probability mass into a **single, sharp 1-frame spike (peak)**, surrounded by long stretches of blank emissions.
- This creates the characteristic "peaky" emission profile of CTC acoustic models.
