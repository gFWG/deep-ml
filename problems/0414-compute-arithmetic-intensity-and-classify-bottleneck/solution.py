def compute_arithmetic_intensity(flops: float, bytes_accessed: float, peak_performance: float, peak_bandwidth: float) -> dict:
    """
    Analyze a computational kernel using the Roofline Model.
    
    Args:
        flops: Total floating-point operations of the kernel
        bytes_accessed: Total bytes transferred to/from memory
        peak_performance: Hardware peak compute throughput (FLOP/s)
        peak_bandwidth: Hardware peak memory bandwidth (bytes/s)
    
    Returns:
        Dictionary with arithmetic_intensity, ridge_point, bottleneck,
        achieved_performance, and utilization_percent
    """
    arithmetic_intensity = round(flops/bytes_accessed, 4)
    ridge_point = round(peak_performance/peak_bandwidth, 4)
    achieved_performance = round(min(arithmetic_intensity*peak_bandwidth, peak_performance), 4)
    return {
        'arithmetic_intensity': arithmetic_intensity, 
        'ridge_point': ridge_point, 
        'bottleneck': 'memory-bound' if arithmetic_intensity < ridge_point else 'compute-bound', 
        'achieved_performance': achieved_performance, 
        'utilization_percent': round(achieved_performance/peak_performance*100, 4)
    }