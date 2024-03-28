# Lecture 1 and 2: Markov Models

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
  - **rows sum to 1**: $\sum_{j=1}^{n} a_{ij} = 1$
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

### Learning Markov Models

- Similar to Naive Bayes, Markov models is just counting
- Given $n$ samples/ sequences, we can find:
  - Initial probabilities: $\pi_0(s_i) = \frac{\text{count}(s_i)}{n}$
  - Transition probabilities: $a_{ij} = \pi(s_i| s_j) = \frac{\text{count}(s_i, s_j)}{\text{count}(s_i)} = \frac{\text{count of state i to j}}{\text{count of state i to any state}}$

### n-gram language model

- Markov model for NLP
- `n` in n-gram means $n-1$ previous words are considered
  - e.g. `n=2` (bigram) means consider 1 previous word
  - DIFFERENT from Markov model definition bigram= markov model with `n=1`
- We extend the definition of a "state" to be a sequence of words
  - e.g. "I have read this book" -> bigram states: "I have", "have read", "read this", "this book"
- example: "I have read this book"
  - bigram: $P(\text{this book} | \text{read this})$
  - trigram: $P(\text{read this book} | \text{have read this})$

#### Evaluating Language Models

- Best way is to embed it in an application and measure how much the application improves (**extrinsic evaluation**)
- Often it is expensive to run NLP pipeline
- It is helpful to have a metric to quickly evaluate performance
- Most common **intrinsic evaluation** metric is **perplexity**
  - **Lower perplexity is better** (means better predictor of the words in test set)

#### Perplexity

Let $W = w_1, w_2, ..., w_N$ be a sequences of words.

$$
\text{Perplexity}(W) = P(w_1, w_2, ..., w_N)^{-\frac{1}{N}} \\
= \sqrt[N]{\frac{1}{P(w_1, w_2, ..., w_N)}}
$$

For `n=1` markov model (bigram):

$$P(w_1, w_2, ..., w_N) = \prod_{i=1}^{N} P(w_i|w_{i-1})$$

So...

$$
\text{Perplexity}(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i|w_{i-1})}}
$$

## Applications of Markov Models

### Google PageRank

- **Idea**: The importance of a page is determined by the importance of the pages that link to it.
- **Markov Model**: The probability of being on a page at time $t$ depends only on the page at time $t-1$.
- **Transition Matrix**: The probability of going from page $i$ to page $j$ is the number of links from page $i$ to page $j$ divided by the number of links from page $i$.
  - Add $\epsilon$ to all values so that matrix is fully connected
  - Normalize so sum of each row is 1
- **Stationary Distribution**: The stationary distribution of the transition matrix gives the importance of each page.
  - It shows the page's long-term visit rate

## Basic Text Preprocessing

- Text is unstructured and messy
  - Need to "normalize"

### Tokenization

- Sentence segmentation: text -> sentences
- Word tokenization: sentence -> words
  - Process of identifying word boundaries
- Characters for tokenization:
  | Character | Description |
  | --- | --- |
  | Space | Separate words |
  | dot `.` | Kind of ambiguous (e.g. `U.S.A`) |
  | `!`, `?` | Kind of ambiguous too |
- How?
  - Regex
  - Use libraries like `nltk`, `spacy`, `stanza`

### Word Segmentation

- In NLP we talk about:
  - **Type**: Unique words (element in vocabulary)
  - **Token**: Instances of words

### Other Preprocessing Steps

- Removing stop words
- Lemmatization: Convert words to their base form
- Stemming: Remove suffixes
  - e.g. automates, automatic, automation -> automat
  - Not actual words, but can be useful for some tasks
  - Be careful, because kind of aggressive

### Other Typical NLP Tasks

- **Part of Speech (POS) Tagging**: Assigning a part of speech to each word
- **Named Entity Recognition (NER)**: Identifying named entities in text
- **Coreference Resolution**: Identifying which words refer to the same entity
- **Dependency Parsing**: Identifying the grammatical structure of a sentence
