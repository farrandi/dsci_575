# Lecture 3,4: Hidden Markov Models

## Hidden Markov Models

### Speech Recognition

- Python has several libraries for speech recognition.
  - Have a module called `SpeechRecognition` which can access:
    - Google Web Speech API
    - Sphinx
    - Wit.ai
    - Microsoft Bing Voice Recognition
    - IBM Speech to Text
  - Might need to pay for some of these services
- **General Task**: Given a sequence of audio signals, want to recognize the corresponding phenomes/ words
  - **Phenomes**: Distinct units of sound
    - E.g. "cat" has 3 phenomes: "k", "ae", "t". "dog" has 3 phenomes: "d", "aa", "g"
  - English has ~44 phenomes
- It is a **sequence modeling problem**
- Many modern speech recognition systems use HMM
  - HMM is also still useful in bioinformatics, financial modeling, etc.

### HMM Definition and Example

- **Hidden**: The state is not directly observable
  - e.g. In speech recognition, the phenome is not directly observable. Or POS (Part of Speech) tags in text.
- HMM is specified by a 5-tuple $(S, Y, \pi, T, B)$
  - $S$: Set of states
  - $Y$: Set of observations
  - $\pi$: Initial state probabilities
  - $T$: Transition matrix, where $a_{ij}$ is the probability of transitioning from state $s_i$ to state $s_j$
  - $B$: Emission probabilities. $b_j(y)$ is the probability of observing $y$ in state $s_j$
- Yielding the state sequence and observation sequence

$$\text{State Sequence}:Q = q_1, q_2, ..., q_T \in S$$

$$\text{Observation Sequence}: O = o_1, o_2, ..., o_T \in Y$$

#### HMM Assumptions

<img src="images/3_hmm.png" width="350">

1. The probability of a particular state depends only on the previous state

$$P(q_i|q_0,q_1,\dots,q_{i-1})=P(q_i|q_{i-1})$$

2. Probability of an observation depends **only** on the state.

$$P(o_i|q_0,q_1,\dots,q_{i-1},o_0,o_1,\dots,o_{i-1})=P(o_i|q_i)$$

**Important Notes**:

- Observations are ONLY dependent on the current state
- States are dependent on the previous state (not observations)
- Each hidden state has a probability distribution over all observations

#### Fundamental Questions for a HMM

1. Likelihood
   - Given $\theta = (\pi, T, B)$ what is the probability of observation sequence $O$?
2. Decoding
   - Given an observation sequence $O$ and model $\theta$. How do we choose the best state sequence $Q$?
3. Learning
   - Given an observation sequence $O$, how do we learn the model $\theta = (\pi, T, B)$?

#### HMM Likelihood

- What is the probability of observing sequence $O$?

$$P(O) = \sum\limits_{Q} P(O,Q)$$

This means we need all the possible state sequences $Q$

$$P(O,Q) = P(O|Q)\times P(Q) = \prod\limits_{i=1}^T P(o_i|q_i) \times \prod\limits_{i=1}^T P(q_i|q_{i-1})$$

This is computationally inefficient. $O(2Tn^T)$

- Need to find every possible state sequence $n^T$, then consider each emission given the state sequence $T$
- $n$ is the number of hidden states
- $T$ is the length of the sequence

To solve this, we use dynamic programming (Forward Procedure)

##### Dynamic Programming: Forward Procedure

- Find $P(O|\theta)$
- Make a table of size $n \times T$ called **Trellis**
  - rows: hidden states
  - columns: time steps
- Fill the table using the following formula:
  1. **Initialization**: compute first column ($t=0$)
     - $\alpha_j(0) = \pi_j b_j(o_1)$
       - $\pi_j$: initial state probability
       - $b_j(o_1)$: emission probability
  2. **Induction**: compute the rest of the columns ($1 \leq t < T$)
     - $\alpha_j(t+1) = \sum\limits_{i=1}^n \alpha_i(t) a_{ij} b_j(o_{t+1})$
       - $a_{ij}$: transition probability from $i$ to $j$
  3. **Termination**: sum over the last column ($t=T$)
     - $P(O|\theta) = \sum\limits_{i=1}^n \alpha_T(i)$
- It is computed left to right and top to bottom
- Time complexity: $O(2n^2T)$
  - At each time step, need to compare states to all other states $n^2$
  - Better compared to the naive approach $O(2Tn^T)$

### Supervised Learning in HMM

- Training data: Set of observations $O$ and set of state sequences $Q$
- Find parameters $\theta = (\pi, T, B)$

- Popular libraries in Python:
  - `hmmlearn`
  - `pomegranate`

### Decoding: The Viterbi Algorithm

- Given an observation sequence $O$ and model $\theta = (\pi, T, B)$, how do we choose the best state sequence $Q$?
- Find $Q^* = \arg\max_Q P(O,Q|\theta)$

|                             | Forward Procedure                                                                                                                         | Viterbi Algorithm                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Purpose**                 | Computes the probability of observing a given sequence of emissions, given the model parameters.                                          | Finds the most likely sequence of hidden states that explains the observed sequence of emissions, given the model parameters. |
| **Computation**             | Computes forward probabilities, which are the probabilities of being in a particular state at each time step given the observed sequence. | Computes the most likely sequence of hidden states.                                                                           |
| **Probability Calculation** | Sum over all possible paths through the hidden states.                                                                                    | Recursively calculates the probabilities of the most likely path up to each state at each time step.                          |
| **Objective**               | Computes the likelihood of observing a given sequence of emissions.                                                                       | Finds the most probable sequence of hidden states that explains the observed sequence of emissions.                           |

- Both are dynamic programming algorithms with time complexity $O(n^2T)$

- **Viterbi Overview**:
  - Store $\delta$ and $\psi$ at each node in the trellis
    - $\delta_i(t)$ is the max probability of the most likely path ending in trellis node at state $i$ at time $t$
    - $\psi_i(t)$ is the best possible previous state at time $t-1$ that leads to state $i$ at time $t$

<img src="images/4_viterbi.png" width="400">

#### Viterbi: Initialization

- $\delta_i(0) = \pi_i b_i(O_0)$
  - recall $b_i(O_0)$ is the emission probability and $\pi_i$ is the initial state probability
- $\psi_i(0) = 0$

#### Viterbi: Induction

- Best path $\delta_j(t)$ to state $j$ at time $t$ depends on each previous state and
  their transition to state $j$

- $\delta_j(t) = \max\limits_i \{\delta_i(t-1)a_{ij}\} b_j(o_t)$
  - $b_j(o_t)$ is the emission probability of observation $o_t$ given state $j$
- $\psi_j(t) = \arg \max\limits_i \{\delta_i(t-1)a_{ij}\}$

#### Viterbi: Conclusion

- Choose the best final state
  - $q_t^* = \arg\max\limits_i \delta_i(T)$
- Recursively choose the best previous state
  - $q_{t-1}^* = \psi_{q_t^*}(T)$

### The Backward Procedure

- We do not always have mapping from observations to states (emission probabilities $B$)
- Given an observation sequence $O$ but not the state sequence $Q$, how do we choose the best parameters $\theta = (\pi, T, B)$?
- Use **forward-backward algorithm**

#### Basic Idea

- Reverse of the forward procedure **right to left** but still **top to bottom**
- Find the probability of observing the rest of the sequence given the current state
  - $\beta_j(t) = P(o_{t+1}, o_{t+2}, \dots, o_T)$

#### Steps for Backward Procedure

1. **Initialization**: set all values at last time step to 1
   - $\beta_j(T) = 1$
2. **Induction**: compute the rest of the columns ($1 \leq t < T$)
   - $\beta_i(t) = \sum_{j=1}^N a_{ij}b_{j}(o_{t+1}) \beta_j(t+1)$
3. **Conclusion**: sum over the first column
   - $P(O|\theta) = \sum_{i=1}^N \pi_i b_i(o_1) \beta_i(1)$
