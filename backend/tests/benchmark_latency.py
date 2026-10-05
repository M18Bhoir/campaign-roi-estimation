import json
import statistics
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app


# ============================================================
# Configuration
# ============================================================

WARMUP_REQUESTS = 10
BENCHMARK_REQUESTS = 100

OUTPUT_DIR = Path("backend/results")
OUTPUT_FILE = OUTPUT_DIR / "latency_benchmark.json"


# ============================================================
# Test Client
# ============================================================

client = TestClient(app)


# ============================================================
# Valid Campaign
# ============================================================

VALID_CAMPAIGN = {
    "budget": 40365.77,
    "competitor_score": 59,
    "campaign_duration_days": 72,
    "discount_percent": 0.34,
    "platform": "Twitter (X)",
    "region": "Asia Pacific",
    "device": "Smart TV",
    "customer_segment": "New",
    "product_category": "Real Estate",
    "campaign_type": "Sales",
    "season": "Holiday",
    "marketing_objective": "Sales Growth",
    "campaign_year": 2025,
    "campaign_month": 10,
    "campaign_quarter": 4,
    "campaign_day_of_week": 3,
}


# ============================================================
# Percentile Function
# ============================================================

def calculate_percentile(values, percentile):
    """
    Calculate percentile using linear interpolation.
    """

    if not values:
        return 0.0

    sorted_values = sorted(values)

    position = (len(sorted_values) - 1) * percentile

    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    fraction = position - lower_index

    return (
        sorted_values[lower_index]
        + fraction
        * (
            sorted_values[upper_index]
            - sorted_values[lower_index]
        )
    )


# ============================================================
# Warmup
# ============================================================

def run_warmup():
    print("\n" + "=" * 60)
    print("WARMUP")
    print("=" * 60)

    for i in range(WARMUP_REQUESTS):
        response = client.post(
            "/api/predict",
            json=VALID_CAMPAIGN,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Warmup request {i + 1} failed: "
                f"{response.status_code}"
            )

    print(
        f"Warmup completed: "
        f"{WARMUP_REQUESTS} requests"
    )


# ============================================================
# Benchmark
# ============================================================

def run_benchmark():
    api_latencies = []
    model_latencies = []

    successful_requests = 0
    failed_requests = 0

    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK")
    print("=" * 60)

    for i in range(BENCHMARK_REQUESTS):

        start_time = time.perf_counter()

        response = client.post(
            "/api/predict",
            json=VALID_CAMPAIGN,
        )

        end_time = time.perf_counter()

        api_latency_ms = (
            end_time - start_time
        ) * 1000

        if response.status_code == 200:

            successful_requests += 1

            response_data = response.json()

            model_latency_ms = response_data[
                "data"
            ]["prediction_latency_ms"]

            api_latencies.append(
                api_latency_ms
            )

            model_latencies.append(
                model_latency_ms
            )

        else:
            failed_requests += 1

            print(
                f"Request {i + 1} failed: "
                f"{response.status_code}"
            )

    return (
        api_latencies,
        model_latencies,
        successful_requests,
        failed_requests,
    )


# ============================================================
# Statistics
# ============================================================

def calculate_statistics(values):

    if not values:
        return {
            "min_ms": 0.0,
            "mean_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "std_dev_ms": 0.0,
        }

    return {
        "min_ms": round(min(values), 3),
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(
            calculate_percentile(values, 0.95),
            3,
        ),
        "p99_ms": round(
            calculate_percentile(values, 0.99),
            3,
        ),
        "max_ms": round(max(values), 3),
        "std_dev_ms": round(
            statistics.stdev(values)
            if len(values) > 1
            else 0.0,
            3,
        ),
    }


# ============================================================
# Save Results
# ============================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print(
        f"\nBenchmark results saved to:\n"
        f"{OUTPUT_FILE}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("CAMPAIGN ROI API - LATENCY BENCHMARK")
    print("=" * 60)

    print(
        f"Warmup requests:    {WARMUP_REQUESTS}"
    )

    print(
        f"Benchmark requests: {BENCHMARK_REQUESTS}"
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    run_warmup()

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    (
        api_latencies,
        model_latencies,
        successful_requests,
        failed_requests,
    ) = run_benchmark()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    api_statistics = calculate_statistics(
        api_latencies
    )

    model_statistics = calculate_statistics(
        model_latencies
    )

    success_rate = (
        successful_requests
        / BENCHMARK_REQUESTS
    ) * 100

    results = {
        "benchmark": {
            "warmup_requests": WARMUP_REQUESTS,
            "benchmark_requests": BENCHMARK_REQUESTS,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate_percent": round(
                success_rate,
                3,
            ),
        },
        "api_request_latency_ms": api_statistics,
        "model_prediction_latency_ms": model_statistics,
    }

    # --------------------------------------------------------
    # Console Output
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print("\nAPI REQUEST LATENCY")
    print("-" * 40)

    print(
        f"Minimum       : "
        f"{api_statistics['min_ms']} ms"
    )

    print(
        f"Mean          : "
        f"{api_statistics['mean_ms']} ms"
    )

    print(
        f"Median        : "
        f"{api_statistics['median_ms']} ms"
    )

    print(
        f"P95           : "
        f"{api_statistics['p95_ms']} ms"
    )

    print(
        f"P99           : "
        f"{api_statistics['p99_ms']} ms"
    )

    print(
        f"Maximum       : "
        f"{api_statistics['max_ms']} ms"
    )

    print(
        f"Std Deviation : "
        f"{api_statistics['std_dev_ms']} ms"
    )

    print("\nMODEL PREDICTION LATENCY")
    print("-" * 40)

    print(
        f"Minimum       : "
        f"{model_statistics['min_ms']} ms"
    )

    print(
        f"Mean          : "
        f"{model_statistics['mean_ms']} ms"
    )

    print(
        f"Median        : "
        f"{model_statistics['median_ms']} ms"
    )

    print(
        f"P95           : "
        f"{model_statistics['p95_ms']} ms"
    )

    print(
        f"P99           : "
        f"{model_statistics['p99_ms']} ms"
    )

    print(
        f"Maximum       : "
        f"{model_statistics['max_ms']} ms"
    )

    print(
        f"Std Deviation : "
        f"{model_statistics['std_dev_ms']} ms"
    )

    print("\nREQUEST SUMMARY")
    print("-" * 40)

    print(
        f"Successful    : "
        f"{successful_requests}"
    )

    print(
        f"Failed        : "
        f"{failed_requests}"
    )

    print(
        f"Success Rate  : "
        f"{success_rate:.2f}%"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(results)

    print("\n" + "=" * 60)
    print("LATENCY BENCHMARK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()