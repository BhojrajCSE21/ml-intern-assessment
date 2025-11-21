# Evaluation

## Task 1: Trigram Language Model

### Design Choices

1.  **Data Structure**: I used a nested `defaultdict` (`self.counts = defaultdict(lambda: defaultdict(int))`) to store the n-gram counts. This allows for efficient O(1) access to the counts of the next word given the previous two words. The outer dictionary keys are tuples of two words `(w1, w2)`, and the inner dictionary keys are the next words `w3`.

2.  **Preprocessing**:
    -   **Text Cleaning**: I converted all text to lowercase and removed punctuation using regex (`re.sub(r'[^\w\s]', '', text)`). This simplifies the vocabulary and ensures that "The" and "the" are treated as the same word.
    -   **Tokenization**: I used simple whitespace splitting (`text.split()`).
    -   **Padding**: Since it's a trigram model, I padded the start of the sequence with two `<START>` tokens so that the first word in the text can be predicted from `(<START>, <START>)`. I added one `<END>` token at the end to mark the end of the sequence.

3.  **Generation**:
    -   The generation starts with `['<START>', '<START>']`.
    -   At each step, I look up the counts for the last two words.
    -   I convert the counts to probabilities implicitly by using `random.choices` with `weights=counts`. This performs weighted random sampling, ensuring that words with higher counts are more likely to be chosen.
    -   Generation stops when `<END>` is sampled or the maximum length is reached.

### Verification
-   **Tests**: I ran the provided tests in `ml-assignment/tests/test_ngram.py` using `pytest`. All tests passed.
-   **Manual Check**: I verified the generation logic by inspecting the code and running the tests.

## Task 2: Scaled Dot-Product Attention (Optional)

### Implementation
I implemented the Scaled Dot-Product Attention mechanism using `numpy` as requested.

The function `scaled_dot_product_attention(Q, K, V, mask=None)` performs the following steps:
1.  **Score Calculation**: Computes $scores = \frac{QK^T}{\sqrt{d_k}}$.
2.  **Masking**: If a mask is provided, it sets the scores of masked positions to a very large negative number (`-1e9`) so that their probability becomes 0 after softmax.
3.  **Softmax**: Applies softmax to the scores to get attention weights.
4.  **Weighted Sum**: Computes the output as the weighted sum of values: $output = weights \cdot V$.

### Verification
I created a demo script `ml-assignment/task2/demo_attention.py` that:
1.  Generates random Q, K, V matrices.
2.  Computes attention without a mask.
3.  Computes attention with a mask (masking the last key).
4.  Verifies that the weights for masked positions are 0.

The demo output confirms that the implementation works correctly.
