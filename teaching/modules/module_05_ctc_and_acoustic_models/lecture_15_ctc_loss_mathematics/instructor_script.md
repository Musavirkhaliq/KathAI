# Instructor Delivery Script: Lecture 15 (CTC Loss Mathematics)

## Timetable (90 Minutes)
```
00:00 - 00:20 | The Extended Target Lattice (2U + 1 States)
00:20 - 00:50 | Step-by-Step Derivation of Forward Variable Alpha Recursion
00:50 - 01:15 | The Elegant Gradient: (y_k^t - gamma_t(k))
01:15 - 01:25 | Explaining Peaky Behavior: Why CTC Loves the Blank Token
01:25 - 01:30 | Summary & Homework Assignment
```

---

## Instructor Notes & Dialogue

### 00:20 - 00:50 | Forward Recursion on the Board
- Draw a 2D grid: Time frames $t=1 \dots T$ horizontally, Extended states $s=1 \dots 2U+1$ vertically.
- Mark the states for `"كٲشُر"`:
  - $s=1: \epsilon$
  - `s=2: "ك"`
  - $s=3: \epsilon$
  - `s=4: "ٲ"`
  - $s=5: \epsilon$
  - `s=6: "ش"`
  - $s=7: \epsilon$
  - `s=8: "ر"`
  - $s=9: \epsilon$
- Trace the three incoming arrows into state 4 (`ٲ`):
  1. Horizontal arrow: Stayed at `ٲ` ($\alpha_{t-1}(4)$).
  2. Diagonal arrow 1: Came from intermediate blank $\epsilon$ ($\alpha_{t-1}(3)$).
  3. Diagonal arrow 2: **Skipped the blank completely** from `ك` ($\alpha_{t-1}(2)$)!
- Point to arrow 3:
  > "Why can you skip the blank here?
  > Because 'ك' and 'ٲ' are different letters! You don't need a blank to separate them.
  > But what if the two letters were identical, like two consecutive Shins?
  > Then arrow 3 is forbidden! You MUST pass through state 3. That is the genius of Alex Graves' formulation."
