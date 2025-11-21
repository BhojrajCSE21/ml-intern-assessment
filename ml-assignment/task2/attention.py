import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Computes the scaled dot-product attention.

    Args:
        Q (numpy.ndarray): Queries matrix of shape (..., seq_len_q, d_k)
        K (numpy.ndarray): Keys matrix of shape (..., seq_len_k, d_k)
        V (numpy.ndarray): Values matrix of shape (..., seq_len_k, d_v)
        mask (numpy.ndarray, optional): Mask tensor of shape (..., seq_len_q, seq_len_k).
                                        Defaults to None.

    Returns:
        tuple:
            - output (numpy.ndarray): The weighted sum of values, shape (..., seq_len_q, d_v)
            - attention_weights (numpy.ndarray): Attention weights, shape (..., seq_len_q, seq_len_k)
    """
    # 1. Matmul Q and K^T
    # Q: (..., seq_len_q, d_k)
    # K: (..., seq_len_k, d_k) -> K^T: (..., d_k, seq_len_k)
    # scores: (..., seq_len_q, seq_len_k)
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.swapaxes(-2, -1)) / np.sqrt(d_k)

    # 2. Apply mask (if provided)
    if mask is not None:
        # Mask should be broadcastable to scores
        # Usually mask has 0 for valid positions and 1 (or large negative value) for masked positions
        # Or boolean mask where False is masked.
        # Let's assume standard transformer mask where 0 indicates "mask this out" (or 1 indicates "mask this out").
        # Actually, standard practice often uses -inf for masked positions before softmax.
        # Let's assume mask is boolean: True means keep, False means mask.
        # Or 1 means keep, 0 means mask.
        # If mask is 0, set score to -inf.
        scores = np.where(mask == 0, -1e9, scores)

    # 3. Softmax
    # Compute softmax along the last dimension (seq_len_k)
    # Subtract max for numerical stability
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 4. Matmul weights and V
    # weights: (..., seq_len_q, seq_len_k)
    # V: (..., seq_len_k, d_v)
    # output: (..., seq_len_q, d_v)
    output = np.matmul(attention_weights, V)

    return output, attention_weights
