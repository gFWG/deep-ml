import torch

def attention_memory_flops(B: int, h: int, N: int, d: int, bytes_per_element: int = 2) -> dict:
    """
    Compute memory traffic and FLOPs for standard self-attention using PyTorch tensors.

    The attention mechanism uses:
    - torch.bmm for Q @ K^T and P @ V matrix multiplications
    - torch.softmax for row-wise normalization

    Args:
        B: Batch size
        h: Number of attention heads
        N: Sequence length
        d: Head dimension
        bytes_per_element: Bytes per element (e.g., 2 for FP16, 4 for FP32)

    Returns:
        dict with keys:
            'qk_flops': int - FLOPs for Q @ K^T
            'softmax_flops': int - FLOPs for softmax
            'pv_flops': int - FLOPs for P @ V
            'total_flops': int - Total FLOPs
            'memory_bytes': int - Total memory traffic in bytes
            'arithmetic_intensity': float - FLOPs per byte, rounded to 2 decimal places
    """
    # Your code here

    # Total number of independently processed attention heads
    batch_heads = B * h

    # 1. Q @ K^T: (N, d) @ (d, N) -> (N, N)
    qk_flops = 2 * batch_heads * N * N * d

    # 2. Row-wise softmax over the attention scores
    softmax_flops = 5 * batch_heads * N * N

    # 3. P @ V: (N, N) @ (N, d) -> (N, d)
    pv_flops = 2 * batch_heads * N * N * d

    # 4. Total computation
    total_flops = qk_flops + softmax_flops + pv_flops

    # 5. Memory traffic for unfused attention
    # Q, K, V: three input tensors
    # O: one output tensor
    qkv_output_elements = 4 * batch_heads * N * d

    # Scores S and probabilities P:
    # each written once and read once
    attention_elements = 4 * batch_heads * N * N

    memory_bytes = (
        qkv_output_elements + attention_elements
    ) * bytes_per_element

    # 6. Arithmetic intensity (FLOPs per byte)
    arithmetic_intensity = round(
        total_flops / memory_bytes, 2
    )

    return {
        "qk_flops": qk_flops,
        "softmax_flops": softmax_flops,
        "pv_flops": pv_flops,
        "total_flops": total_flops,
        "memory_bytes": memory_bytes,
        "arithmetic_intensity": arithmetic_intensity,
    }