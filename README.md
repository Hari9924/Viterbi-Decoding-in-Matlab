# Viterbi-Decoding-in-Matlab
###**Viterbi Decoding**
Viterbi decoding is an algorithm used to decode signals that have been encoded using convolutional codes. It is a maximum likelihood decoding technique that finds the most probable transmitted sequence based on the received noisy sequence. The algorithm is widely used in error correction for digital communication systems, including wireless networks, deep-space communications, and trellis-based ISI channels.

###**The algorithm works in three main steps:**

**Branch Metric Calculation**

The received signal is compared with all possible transmitted symbols, and a branch metric (distance or likelihood) is calculated.
**Path Metric Calculation & State Transition (Trellis Diagram)**

The decoder maintains multiple paths through the trellis diagram, updating the path metrics at each stage by choosing the most likely path.
**Traceback and Decision**

After processing the entire sequence, the decoder traces back the most likely path from the last state to determine the decoded output.

**Applications of Viterbi Decoding**
-Error Correction in Communication Systems (e.g., GSM, LTE, deep-space communication).
-Turbo Codes and Trellis-Coded Modulation (TCM).
-Speech and Pattern Recognition (e.g., Hidden Markov Models).
-Decoding in ISI (Inter-Symbol Interference) Channels.
