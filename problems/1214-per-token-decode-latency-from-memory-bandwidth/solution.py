def estimate_decode_latency(num_params, bytes_per_param, bandwidth_bytes_per_s):
    # Return [latency_ms, tokens_per_sec]
    x = num_params/bandwidth_bytes_per_s*bytes_per_param
    return [x*1000, 1/x]