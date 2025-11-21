import numpy as np
from attention import scaled_dot_product_attention

def demo():
    print("--- Scaled Dot-Product Attention Demo ---")
    
    # Define dimensions
    batch_size = 1
    seq_len_q = 3
    seq_len_k = 3
    d_k = 4
    d_v = 4
    
    np.random.seed(42)
    
    # Create random Q, K, V matrices
    Q = np.random.rand(batch_size, seq_len_q, d_k)
    K = np.random.rand(batch_size, seq_len_k, d_k)
    V = np.random.rand(batch_size, seq_len_k, d_v)
    
    print(f"Query shape: {Q.shape}")
    print(f"Key shape: {K.shape}")
    print(f"Value shape: {V.shape}")
    
    # 1. Basic Attention
    print("\n1. Basic Attention (No Mask)")
    output, weights = scaled_dot_product_attention(Q, K, V)
    print("Output shape:", output.shape)
    print("Attention Weights:\n", weights)
    
    # 2. Masked Attention
    print("\n2. Masked Attention")
    # Mask out the last key for all queries
    mask = np.array([[[1, 1, 0]]]) # Shape (1, 1, 3) - broadcastable to (1, 3, 3)
    print("Mask shape:", mask.shape)
    print("Mask:\n", mask)
    
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)
    print("Output shape:", output_masked.shape)
    print("Masked Attention Weights:\n", weights_masked)
    
    # Check if the last column of weights is 0
    assert np.allclose(weights_masked[:, :, -1], 0), "Masked positions should have 0 weight"
    print("\nVerification: Masked positions have 0 weight (verified).")

if __name__ == "__main__":
    demo()
