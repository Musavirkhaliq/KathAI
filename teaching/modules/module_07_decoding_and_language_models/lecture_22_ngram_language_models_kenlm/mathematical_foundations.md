# Mathematical Foundations: Modified Kneser-Ney Discount Formulations

---

## 1. Closed-Form Estimates for Modified Kneser-Ney Discounts

Let $n_k$ denote the total number of $n$-grams with count $k$ in the training corpus:

$$Y = \frac{n_1}{n_1 + 2 n_2}$$

The three discount parameters are computed as:

$$D_1 = 1 - 2 Y \frac{n_2}{n_1}$$
$$D_2 = 2 - 3 Y \frac{n_3}{n_2}$$
$$D_3 = 3 - 4 Y \frac{n_4}{n_3}$$

The backoff weight $\gamma(w_{i-n+1}^{i-1})$ is:

$$\gamma(w_{i-n+1}^{i-1}) = \frac{D_1 N_1(w_{i-n+1}^{i-1}) + D_2 N_2(w_{i-n+1}^{i-1}) + D_3 N_{3+}(w_{i-n+1}^{i-1})}{C(w_{i-n+1}^{i-1})}$$

where $N_k(h) = |\{ w : C(h, w) = k \}|$ is the number of distinct words following context history $h$ with frequency $k$.
