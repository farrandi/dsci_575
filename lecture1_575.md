# Lecture 1: Markov Models

## Language Models

- It computes the probability distribution of a sequence of words.
  - $P(w_1, w_2, ..., w_t)$
  - E.g. P("I have read this book) > P("Eye have red this book")
- Can also get the probability of the upcoming word.
  - $P(w_t | w_1, w_2, ..., w_{t-1})$
  - E.g. P("book" | "I have read this") > P("book" | "I have red this")

### Large Language Models

- Large language models are trained on a large corpus of text.

## Markov Model

- **High-level**: The probability of a word depends only on the previous word (forget everything written before that).
- **Idea**: Predict future depending upon:
  - The current state
  - The probability of change

### Markov Assumption

Naive probability of a sequence of words:
$$P(w_1, w_2, ..., w_t) = P(w_1)P(w_2|w_1)P(w_3|w_1, w_2)...P(w_t|w_1, w_2, ..., w_{t-1})$$

e.g. $$P(\text{I have read this book}) = P(\text{I})P(\text{have}|\text{I})P(\text{read}|\text{I have})P(\text{this}|\text{I have read})P(\text{book}|\text{I have read this})$$

Or simply:
$$P(w_1, w_2, ..., w_t) = \prod_{i=1}^{t} P(w_i|w_{1:i-1})$$

But this is hard, so in Markov model (n-grams), we only consider the `n` previous words. With the assumption:

$$P(w_t|w_1, w_2, ..., w_{t-1}) \approx P(w_t| w_{t-1})$$

### Markov Chain Definition

- Have a set of states $S = \{s_1, s_2, ..., s_n\}$.
- A set of discrete initial probabilities $\pi_0 = \{\pi_0(s_1), \pi_0(s_2), ..., \pi_0(s_n)\}$.
- A transition matrix $T$ where each $a_{ij}$ is the probability of transitioning from state $s_i$ to state $s_j$.

$$
T =
\begin{bmatrix}
    a_{11}       & a_{12} & a_{13} & \dots & a_{1n} \\
    a_{21}       & a_{22} & a_{23} & \dots & a_{2n} \\
    \dots \\
    a_{n1}       & a_{n2} & a_{n3} & \dots & a_{nn}
\end{bmatrix}
$$

- **Properties**:
  - $0 \leq a_{ij} \leq 1$
  - rows sum to 1: $\sum_{j=1}^{n} a_{ij} = 1$
  - columns do not need to sum to 1
  - This is assuming **Homogeneous Markov chain** (transition matrix does not change over time).

### Markov Chain Tasks

1. Predict probabilities of sequences of states
2. Compute probability of being at a state at a given time
3. Stationary Distribution: Find steady state after a long time
4. Generation: Generate a sequences that follows the probability of states

#### Stationary Distribution

- Steady state after a long time.
- Basically the eigenvector of the transition matrix corresponding to the eigenvalue 1.

$$\pi T = \pi$$

- Where $\pi$ is the stationary probability distribution
  </br>
- **Sufficient Condition for Uniqueness**:
  - Positive transitions ($a_{ij} > 0$ for all $i, j$)
- **Weaker Condition for Uniqueness**:
  - **Irreducible**: Can go from any state to any other state (fully connected)
  - **Aperiodic**: No fixed period (does not fall into a repetitive loop)
