# Solutions: Lecture 33 (Production Streaming and Chunking)

---

## Solution to Exercise 1
1. Number of chunks:
   $$\text{Hop } H = 7.0 \text{ s}$$.
   Starts at $0, 7, 14, 21, 28, 35, 42 \implies \mathbf{7 \text{ total chunks}}$.
2. Timestamps:
   - Chunk 1: $[0.0\text{s}, 8.0\text{s}]$
   - Chunk 2: $[7.0\text{s}, 15.0\text{s}]$
   - Chunk 7 (Final): $[42.0\text{s}, 45.0\text{s}]$ (padded to 8.0s).

---

## Solution to Exercise 2
1. RTF:
   $$\text{Duration} = 20 \times 60 = 1200 \text{ seconds}$$
   $$\text{Elapsed} = 60 + 48 = 108 \text{ seconds}$$
   $$\text{RTF} = \frac{108}{1200} = \mathbf{0.090}$$
2. Concurrent streams:
   $$\text{Streams} \le \frac{1.0}{\text{RTF}} = \frac{1.0}{0.090} \approx \mathbf{11 \text{ concurrent real-time streams}}$$
