import numpy as np

def decompose_latency(stage_latencies: dict, percentiles: list) -> dict:
    """
    Decompose end-to-end inference latency into component stages.
    
    Args:
        stage_latencies: dict mapping stage name -> np.ndarray of latency measurements (ms)
        percentiles: list of percentile values to compute (e.g., [50, 95, 99])
    
    Returns:
        Dictionary with keys: 'e2e_mean', 'e2e_percentiles', 'stage_stats',
                               'bottleneck', 'stage_pct'
    """
    # Step 1: Calculate end-to-end latency for every request
    e2e = np.sum(list(stage_latencies.values()), axis=0)

    # Step 2: Calculate end-to-end statistics
    e2e_mean = round(float(np.mean(e2e)), 2)

    e2e_percentiles = {
        p: round(float(np.percentile(e2e, p)), 2)
        for p in percentiles
    }

    # Step 3: Calculate statistics for each stage
    stage_stats = {}

    for name, latencies in stage_latencies.items():
        stage_stats[name] = {
            "mean": round(float(np.mean(latencies)), 2),
            "std": round(float(np.std(latencies, ddof=0)), 2),
            "percentiles": {
                p: round(float(np.percentile(latencies, p)), 2)
                for p in percentiles
            }
        }

    # Step 4: Identify the bottleneck using unrounded means
    bottleneck = max(
        stage_latencies,
        key=lambda name: np.mean(stage_latencies[name])
    )

    # Step 5: Calculate each stage's contribution
    total_mean = float(np.mean(e2e))

    stage_pct = {
        name: round(float(np.mean(latencies)) / total_mean * 100, 2)
        for name, latencies in stage_latencies.items()
    }

    # Step 6: Return the report
    return {
        "e2e_mean": e2e_mean,
        "e2e_percentiles": e2e_percentiles,
        "stage_stats": stage_stats,
        "bottleneck": bottleneck,
        "stage_pct": stage_pct
    }