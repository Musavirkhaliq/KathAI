# Exercises: Lecture 33 (Production Streaming and Chunking)

---

## Exercise 1: Chunk Slicing Geometry
An audio recording has a duration of 45.0 seconds at 16,000 Hz.
Using Method C2 with $W = 8.0 \text{ s}$ and $O = 1.0 \text{ s}$ (hop $H = 7.0 \text{ s}$):
1. How many total chunks are generated?
2. What are the start and end timestamps (in seconds) for Chunk 1, Chunk 2, and the final Chunk?

---

## Exercise 2: Real-Time Factor Calculation
A production server processes a 20-minute radio broadcast in 1 minute and 48 seconds.
1. What is the Real-Time Factor (RTF)?
2. How many simultaneous real-time speech streams can this server support concurrently on a single GPU?
