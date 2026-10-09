def classify_llm_phases(num_params: int, sequence_length: int, batch_size: int, bytes_per_param: int, peak_flops: float, peak_bandwidth: float) -> dict:
    """
    Analyze prefill and decode phases of LLM inference using the Roofline Model.

    Args:
        num_params: Total number of model parameters
        sequence_length: Number of input tokens processed during prefill
        batch_size: Number of sequences processed in parallel during decode
        bytes_per_param: Memory footprint per parameter (e.g., 2 for FP16)
        peak_flops: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)

    Returns:
        Dictionary containing ridge_point and analysis dicts for 'prefill' and 'decode',
        each with total_flops, memory_bytes, arithmetic_intensity, bottleneck,
        achieved_flops, and utilization_percent.
    """
    ridge_point = round(peak_flops/peak_bandwidth, 4)
    ai1 = round(2*sequence_length/bytes_per_param, 4)
    ai2 = round(2*batch_size/bytes_per_param, 4)
    return {'ridge_point': ridge_point, 

    'prefill': {
        'total_flops': 2*num_params*sequence_length, 
        'memory_bytes': bytes_per_param*num_params, 'arithmetic_intensity': ai1, 
        'bottleneck': 'memory-bound' if ai1 < ridge_point else 'compute-bound', 
        'achieved_flops': round(min(ai1*peak_bandwidth,peak_flops),4), 
        'utilization_percent': round(min(ai1*peak_bandwidth,peak_flops)/peak_flops*100,4), 
    }, 

    'decode': {
        'total_flops': 2*num_params*batch_size, 
        'memory_bytes': bytes_per_param*num_params, 'arithmetic_intensity': ai2, 
        'bottleneck': 'memory-bound' if ai2 < ridge_point else 'compute-bound', 
        'achieved_flops': round(min(ai2*peak_bandwidth,peak_flops),4), 
        'utilization_percent': round(min(ai2*peak_bandwidth,peak_flops)/peak_flops*100,4), 
    }
    }