def estimate_kv_cache_size(model_config: dict, batch_size: int, seq_len: int) -> dict:
    """
    Estimate the KV cache memory footprint for a Transformer model.

    Args:
        model_config: Dictionary with model architecture parameters
        batch_size: Number of sequences in the batch
        seq_len: Number of cached tokens

    Returns:
        Dictionary with cache size estimates
    """
    # Your code here
    num_layers = model_config['num_layers']
    num_attention_heads = model_config['num_attention_heads']
    num_kv_heads = model_config.get('num_kv_heads', num_attention_heads)
    hidden_size = model_config['hidden_size']
    dtype_bytes = model_config['dtype_bytes']

    kv_cache_elements = 2*batch_size*seq_len*num_kv_heads*hidden_size/num_attention_heads*num_layers
    kv_cache_size_bytes = kv_cache_elements*dtype_bytes
    return {
        'kv_cache_elements': int(kv_cache_elements),
        'kv_cache_size_bytes': int(kv_cache_size_bytes),
        'kv_cache_size_mb': round(kv_cache_size_bytes/1024/1024, 4),
        'per_layer_size_mb': round(kv_cache_size_bytes/num_layers/1024/1024, 4),
        'per_token_size_kb': round(2*num_kv_heads*hidden_size/num_attention_heads*num_layers*dtype_bytes/1024, 4)

    }