# Lecture 5: Topic Modeling

## Topic Modeling

- **Motivation**:
  - Humans are good at identifying topics in documents.
  - But, it is difficult to do this at scale. (e.g., 1000s of documents)

### How to do Topic Modeling?

- Common to use unsupervised learning techniques
  - Given hyperparameter $K$, we want to find $K$ topics.
- In unsupervised, a common model:
  - Input:
    - $D$ documents
    - $K$ topics
  - Output:
    - Topic-word association: for each topic, what words describe that topic?
    - Document-topic association: for each document, what topics are in that document?
- Common approaches:
  1. **Latent Semantic Analysis (LSA)**
  2. **Latent Dirichlet Allocation (LDA)**

### Latent Semantic Analysis (LSA)

- Singular Value Decomposition (SVD) of the term-document matrix. See [LSA notes from 563](https://mds.farrandi.com/block_5/563_unsup_learn/563_unsup_learn#lsa-latent-semantic-analysis).

$$X_{n \times d} \approx Z_{n \times k}W_{k \times d}$$

- $n$: number of documents, $d$: number of words, $k$: number of topics

[TODO: Add more notes on LSA]

### Latent Dirichlet Allocation (LDA)

- Bayesian, probabilistic, and generative approach
- Developed by [David Blei](https://www.cs.columbia.edu/~blei/) and colleagues in 2003
  - One of the most cited papers in computer science
- **Basic idea**:
  - Get prior on topic proportions $\theta$
  - Then generate a document with $d$ words:
    - For each word $w_i$:
      - Choose a topic $z_i$ from $\theta$/ **document-topic distribution**
      - Choose a word $w_i$ from the topic $z_i$/ **topic-word distribution**

### Topic Modeling in Python
