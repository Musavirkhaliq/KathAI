# Solutions: Lecture 01 (What is Automatic Speech Recognition?)

---

## Solution to Exercise 1

### Part 1:
The marginal likelihood (evidence) is:

$$P(\mathbf{X}) = \sum_{\mathbf{w}' \in \mathcal{W}} P(\mathbf{X}, \mathbf{w}') = \sum_{\mathbf{w}' \in \mathcal{W}} P(\mathbf{X} \mid \mathbf{w}') P(\mathbf{w}')$$

### Part 2:
In the summation above, $\mathbf{w}'$ is a bound dummy summation variable integrating over all candidate sequences in the vocabulary space $\mathcal{W}$. The expression evaluates to a scalar dependent solely on $\mathbf{X}$. It has no functional dependence on any specific candidate sequence $\mathbf{w}$. Hence:

$$\frac{\partial P(\mathbf{X})}{\partial \mathbf{w}} = 0$$

### Part 3:
Let $\hat{\mathbf{w}}^* = \arg\max_{\mathbf{w} \in \mathcal{W}} [g(\mathbf{X}) P(\mathbf{w} \mid \mathbf{X})]$.
By definition of argmax:

$$g(\mathbf{X}) P(\hat{\mathbf{w}}^* \mid \mathbf{X}) \ge g(\mathbf{X}) P(\mathbf{w} \mid \mathbf{X}) \quad \forall \mathbf{w} \in \mathcal{W}$$

Since $g(\mathbf{X}) > 0$, we can divide both sides by $g(\mathbf{X})$ without altering the direction of the inequality:

$$P(\hat{\mathbf{w}}^* \mid \mathbf{X}) \ge P(\mathbf{w} \mid \mathbf{X}) \quad \forall \mathbf{w} \in \mathcal{W}$$

This is the exact definition of $\hat{\mathbf{w}}^* = \arg\max_{\mathbf{w} \in \mathcal{W}} P(\mathbf{w} \mid \mathbf{X})$. $\blacksquare$

---

## Solution to Exercise 2

### Part 1:
$$L_{\text{emergency}}(\mathbf{w}, \hat{\mathbf{w}}) = \begin{cases}

0 & \text{if } \mathbf{w} = \hat{\mathbf{w}} \\
100 & \text{if } \mathbf{w} = \text{"خَطَر"} \land \hat{\mathbf{w}} \neq \text{"خَطَر"} \\
1 & \text{otherwise}
\end{cases}

$$

### Part 2:
MAP decoding selects the hypothesis with highest posterior probability:
> P(**"خَبَر"** \mid \mathbf{X}) = 0.95 > P(**"خَطَر"** \mid \mathbf{X}) = 0.05 \implies \hat{\mathbf{w}}_{\text{MAP}} = \mathbf{**"خَبَر"**}

### Part 3:
Calculate conditional Bayes risk:
1. If predicting `\hat{\mathbf{w}} = "خَبَر"`:
> R(**"خَبَر"** \mid \mathbf{X}) = L(**"خَطَر"**, **"خَبَر"**) P(**"خَطَر"**) + L(**"خَبَر"**, **"خَبَر"**) P(**"خَبَر"**) = 100 \times 0.05 + 0 \times 0.95 = \mathbf{5.0}
2. If predicting `\hat{\mathbf{w}} = "خَطَر"`:
> R(**"خَطَر"** \mid \mathbf{X}) = L(**"خَطَر"**, **"خَطَر"**) P(**"خَطَر"**) + L(**"خَبَر"**, **"خَطَر"**) P(**"خَبَر"**) = 0 \times 0.05 + 1 \times 0.95 = \mathbf{0.95}
Since $0.95 < 5.0$, MBR decoding selects **`"خَطَر"`**! Even with only 5% probability, the asymmetric penalty for missing a danger alert dictates the optimal decision.

---

## Solution to Exercise 3

### Part 1:
- Vowel A ($F_1 = 300\text{ Hz}, F_2 = 2200\text{ Hz}$): High front unrounded vowel **/i/**.
- Vowel B ($F_1 = 340\text{ Hz}, F_2 = 1450\text{ Hz}$): High central unrounded vowel **/ɨ/** (`إ`).
- Vowel C ($F_1 = 580\text{ Hz}, F_2 = 1550\text{ Hz}$): Low-mid central unrounded vowel **/əː/** (`ٲ`).

### Part 2:
Vowel C corresponds to the letter `ٲ` in `"کٲشُر"` (/kəːʃur/).

### Part 3:
Urdu lacks phonemic central unrounded vowels. Its acoustic vowel space has an empty cavity between $F_2 = 1200\text{ Hz}$ and $F_2 = 1800\text{ Hz}$. When presented with 1450–1550 Hz formant energy, an Urdu acoustic model maps the sound to the nearest available trained Gaussian clusters: front vowels (/i/, /e/) or back vowels (/u/, /o/).

---

## Solution to Exercise 4

### Part 1:
Rejection rate:

$$\frac{191}{500} \times 100 = \mathbf{38.2\% \text{ of utterances rejected}}$$

### Part 2:
Levenshtein WER formula:

$$\text{WER} = \frac{S + D + I}{N} \times 100 = \frac{2118 + 1890 + 412}{5000} \times 100 = \frac{4420}{5000} \times 100 = \mathbf{88.40\%}$$

### Part 3:
The statement is fundamentally flawed because Word Error Rate is an edit distance ratio, not a bounded percentage of correct items. Because insertions can grow unbounded, WER can exceed 100%. A model with 88.4% WER could have recognized only 5% of words correctly if insertions were low, or 30% correctly with high insertions.
