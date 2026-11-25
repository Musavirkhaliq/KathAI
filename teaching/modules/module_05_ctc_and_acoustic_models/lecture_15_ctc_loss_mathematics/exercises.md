# Exercises: Lecture 15 (CTC Loss Mathematics)

---

## Exercise 1: Extended Sequence Construction
For the Kashmiri target word `"سَر"` (Reh, Zain, etc. - 2 characters: $c_1, c_2$):
1. Write the extended target sequence $\mathbf{l}'$. What is $U'$?
2. If the target word has two identical letters `"دَد"` ($c_1, c_1$), explain why $\alpha_t(4)$ cannot receive a transition directly from $\alpha_{t-1}(2)$.

---

## Exercise 2: Peaky Gradient Verification
Suppose at frame $t$, the model's Softmax output for letter 'ب' is `y_ب^t = 0.95`, while the CTC posterior `\gamma_t(ب) = 0.98`.
1. Calculate the logit gradient `\frac{\partial \mathcal{L}}{\partial z_ب^t}`.
2. In which direction will the logit `z_ب^t` move after gradient descent?
