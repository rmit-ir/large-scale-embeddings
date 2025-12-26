#!/usr/bin/env python3
"""
Load test script for search API.
Reads queries from TSV file and sends batch requests to measure latency.
"""

import argparse
import csv
import time
import statistics
import re
from typing import List, Dict, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_server_timing(header_value: str) -> Dict[str, Optional[float]]:
    """Parse server-timing header to extract timing metrics.

    Example: 'embed;dur=75.19, search;dur=32.26, expand;dur=9.14, total;dur=116.60'
    Returns: {'embed': 75.19, 'search': 32.26, 'expand': 9.14, 'total': 116.60}
    """
    timings = {}
    if not header_value:
        return timings

    # Parse each timing component
    pattern = r'(\w+);dur=([\d.]+)'
    matches = re.findall(pattern, header_value)

    for name, duration in matches:
        timings[name] = float(duration)

    return timings


def load_queries_from_tsv(input_file: str) -> List[str]:
    """Load queries from the first column of a TSV file."""
    queries = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row:  # Skip empty rows
                queries.append(row[0])
    return queries


def send_search_request(query: str, url: str, k: int = 100, complexity: int = 50) -> Dict:
    """Send a single search request and measure latency."""
    payload = {
        "query": query,
        "k": k,
        "complexity": complexity
    }

    start_time = time.time()
    try:
        response = requests.post(
            url,
            json=payload,
            headers={
                'accept': 'application/json',
                'Content-Type': 'application/json'
            },
            timeout=30
        )
        latency = time.time() - start_time

        # Parse server-timing header
        server_timing_header = response.headers.get('server-timing', '')
        timings = parse_server_timing(server_timing_header)

        return {
            'query': query,
            'latency': latency,
            'status_code': response.status_code,
            'success': response.status_code == 200,
            'error': None,
            'timings': timings
        }
    except Exception as e:
        latency = time.time() - start_time
        return {
            'query': query,
            'latency': latency,
            'status_code': None,
            'success': False,
            'error': str(e),
            'timings': {}
        }


def batch_search(queries: List[str], url: str, batch_size: int, k: int, complexity: int) -> List[Dict]:
    """Send search requests in batches using thread pool."""
    results = []

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(send_search_request, query, url, k, complexity): query
            for query in queries
        }

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)

            if i % 10 == 0:
                print(f"Completed {i}/{len(queries)} queries")

    return results


def print_statistics(results: List[Dict]):
    """Print latency statistics."""
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    if not successful:
        print("\n❌ All requests failed!")
        for result in failed[:5]:  # Show first 5 errors
            print(f"  Error: {result['error']}")
        return

    latencies = [r['latency'] for r in successful]

    # Collect server-timing metrics
    timing_metrics = {}
    for metric_name in ['embed', 'search', 'expand', 'total']:
        values = [r['timings'].get(
            metric_name) for r in successful if metric_name in r.get('timings', {})]
        if values:
            timing_metrics[metric_name] = values

    print("\n" + "="*60)
    print("SEARCH LATENCY STATISTICS")
    print("="*60)

    print(f"\nTotal Queries:     {len(results)}")
    print(
        f"Successful:        {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(
        f"Failed:            {len(failed)} ({len(failed)/len(results)*100:.1f}%)")

    print(f"\nEnd-to-End Latency Statistics (seconds):")
    print(f"  Mean:            {statistics.mean(latencies):.4f}")
    print(f"  Median:          {statistics.median(latencies):.4f}")
    print(f"  Min:             {min(latencies):.4f}")
    print(f"  Max:             {max(latencies):.4f}")
    print(f"  Std Dev:         {statistics.stdev(latencies):.4f}" if len(
        latencies) > 1 else "  Std Dev:         N/A")

    # Calculate percentiles
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[int(len(sorted_latencies) * 0.50)]
    p90 = sorted_latencies[int(len(sorted_latencies) * 0.90)]
    p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
    p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)
                           ] if len(sorted_latencies) > 100 else sorted_latencies[-1]

    print(f"\nEnd-to-End Percentiles (seconds):")
    print(f"  P50:             {p50:.4f}")
    print(f"  P90:             {p90:.4f}")
    print(f"  P95:             {p95:.4f}")
    print(f"  P99:             {p99:.4f}")

    # Print server-timing statistics
    if timing_metrics:
        print(f"\nServer-Timing Component Statistics (milliseconds):")
        for metric_name in ['embed', 'search', 'expand', 'total']:
            if metric_name in timing_metrics:
                values = timing_metrics[metric_name]
                print(f"\n  {metric_name.capitalize()}:")
                print(f"    Mean:          {statistics.mean(values):.2f}")
                print(f"    Median:        {statistics.median(values):.2f}")
                print(f"    Min:           {min(values):.2f}")
                print(f"    Max:           {max(values):.2f}")
                if len(values) > 1:
                    print(f"    Std Dev:       {statistics.stdev(values):.2f}")

                # Percentiles for this metric
                sorted_values = sorted(values)
                p50_val = sorted_values[int(len(sorted_values) * 0.50)]
                p90_val = sorted_values[int(len(sorted_values) * 0.90)]
                p95_val = sorted_values[int(len(sorted_values) * 0.95)]
                print(f"    P50:           {p50_val:.2f}")
                print(f"    P90:           {p90_val:.2f}")
                print(f"    P95:           {p95_val:.2f}")

    if failed:
        print(f"\nFailed Requests:")
        error_types = {}
        for result in failed:
            error = result['error'] or f"HTTP {result['status_code']}"
            error_types[error] = error_types.get(error, 0) + 1

        for error, count in error_types.items():
            print(f"  {error}: {count}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Load test search API with queries from TSV')
    parser.add_argument('--input', required=True,
                        help='Input TSV file with queries in first column')
    parser.add_argument('--url', default='https://search-cw22b-diskann-minicpm.rankun.org/search',
                        help='Search API endpoint URL')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of concurrent requests (default: 10)')
    parser.add_argument('--k', type=int, default=100,
                        help='Number of results to retrieve (default: 100)')
    parser.add_argument('--complexity', type=int, default=50,
                        help='Search complexity parameter (default: 50)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of queries to test (default: all)')

    args = parser.parse_args()

    print(f"Loading queries from {args.input}...")
    queries = load_queries_from_tsv(args.input)

    if args.limit:
        queries = queries[:args.limit]

    print(f"Loaded {len(queries)} queries")
    print(f"Search API: {args.url}")
    print(f"Batch size: {args.batch_size}")
    print(f"Parameters: k={args.k}, complexity={args.complexity}")
    print(f"\nStarting load test...")

    start_time = time.time()
    results = batch_search(
        queries, args.url, args.batch_size, args.k, args.complexity)
    total_time = time.time() - start_time

    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Throughput: {len(queries)/total_time:.2f} queries/second")

    print_statistics(results)


if __name__ == '__main__':
    main()
