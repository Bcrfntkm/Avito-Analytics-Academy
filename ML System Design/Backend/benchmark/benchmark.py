import asyncio
import argparse
import json
import time
import threading
import statistics
import sys
from typing import List, Dict, Any

import httpx
import psutil
import numpy as np


SAMPLE_TEXTS = [
    "Привет, как дела?",
    "Машинное обучение — это подраздел искусственного интеллекта.",
    "Сегодня хорошая погода для прогулки в парке.",
    "Нейронные сети используются для решения задач классификации и регрессии.",
    "FastAPI — современный веб-фреймворк для создания API на Python.",
    "Трансформеры произвели революцию в обработке естественного языка.",
    "Эмбеддинги позволяют представить текст в виде числового вектора.",
    "Инференс-сервис должен обрабатывать запросы с минимальной задержкой.",
    "Докер упрощает развёртывание приложений в контейнерах.",
    "Метрики производительности помогают оценить качество сервиса.",
]


class ResourceMonitor:
    """Samples CPU and memory usage in a background thread."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.cpu_samples: List[float] = []
        self.mem_samples: List[float] = []  # RSS in MB
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def _run(self) -> None:
        proc = psutil.Process()
        while not self._stop_event.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.mem_samples.append(proc.memory_info().rss / 1024 / 1024)
            time.sleep(self.interval)

    @property
    def avg_cpu(self) -> float:
        return statistics.mean(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def max_cpu(self) -> float:
        return max(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def avg_mem_mb(self) -> float:
        return statistics.mean(self.mem_samples) if self.mem_samples else 0.0

    @property
    def max_mem_mb(self) -> float:
        return max(self.mem_samples) if self.mem_samples else 0.0


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    text: str,
) -> Dict[str, Any]:
    """Send a single /embed request and return timing + inference info."""
    payload = {"text": text}
    t0 = time.perf_counter()
    try:
        response = await client.post(f"{url}/embed", json=payload, timeout=30.0)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        response.raise_for_status()
        data = response.json()
        return {
            "success": True,
            "latency_ms": latency_ms,
            "inference_ms": data.get("processing_time_ms", 0.0),
            "status_code": response.status_code,
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "latency_ms": latency_ms,
            "inference_ms": 0.0,
            "status_code": 0,
            "error": str(exc),
        }


async def run_load_test(
    url: str,
    total_requests: int,
    concurrency: int,
) -> Dict[str, Any]:
    """Run concurrent load test and collect metrics."""
    semaphore = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    async def bounded_request(text: str) -> None:
        async with semaphore:
            result = await send_request(client, url, text)
            results.append(result)

    texts = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(total_requests)]

    monitor = ResourceMonitor(interval=0.5)
    monitor.start()

    async with httpx.AsyncClient() as client:
        wall_start = time.perf_counter()
        await asyncio.gather(*[bounded_request(t) for t in texts])
        wall_elapsed = time.perf_counter() - wall_start

    monitor.stop()

    return {
        "results": results,
        "wall_elapsed_s": wall_elapsed,
        "monitor": monitor,
    }


def compute_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    results = data["results"]
    wall_elapsed = data["wall_elapsed_s"]
    monitor: ResourceMonitor = data["monitor"]

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = [r["latency_ms"] for r in successful]
    inferences = [r["inference_ms"] for r in successful]

    def percentile(data: list, p: float) -> float:
        return float(np.percentile(data, p)) if data else 0.0

    rps = len(successful) / wall_elapsed if wall_elapsed > 0 else 0.0

    return {
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "wall_elapsed_s": round(wall_elapsed, 3),
        "rps": round(rps, 2),
        # End-to-end latency
        "latency_p50_ms": round(percentile(latencies, 50), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "latency_p99_ms": round(percentile(latencies, 99), 2),
        "latency_mean_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "latency_min_ms": round(min(latencies), 2) if latencies else 0.0,
        "latency_max_ms": round(max(latencies), 2) if latencies else 0.0,
        # Pure model inference time
        "inference_p50_ms": round(percentile(inferences, 50), 2),
        "inference_p95_ms": round(percentile(inferences, 95), 2),
        "inference_p99_ms": round(percentile(inferences, 99), 2),
        "inference_mean_ms": round(statistics.mean(inferences), 2) if inferences else 0.0,
        # System resources
        "cpu_avg_pct": round(monitor.avg_cpu, 1),
        "cpu_max_pct": round(monitor.max_cpu, 1),
        "mem_avg_mb": round(monitor.avg_mem_mb, 1),
        "mem_max_mb": round(monitor.max_mem_mb, 1),
    }


def print_report(metrics: Dict[str, Any], concurrency: int) -> None:
    print("\n" + "=" * 60)
    print("  rubert-mini-frida Inference Service — Benchmark Report")
    print("=" * 60)
    print(f"  Total requests  : {metrics['total_requests']}")
    print(f"  Successful      : {metrics['successful']}")
    print(f"  Failed          : {metrics['failed']}")
    print(f"  Concurrency     : {concurrency}")
    print(f"  Wall time       : {metrics['wall_elapsed_s']} s")
    print()
    print("  ── Throughput ──────────────────────────────────────────")
    print(f"  RPS             : {metrics['rps']}")
    print()
    print("  ── End-to-End Latency (HTTP) ───────────────────────────")
    print(f"  P50             : {metrics['latency_p50_ms']} ms")
    print(f"  P95             : {metrics['latency_p95_ms']} ms")
    print(f"  P99             : {metrics['latency_p99_ms']} ms")
    print(f"  Mean            : {metrics['latency_mean_ms']} ms")
    print(f"  Min             : {metrics['latency_min_ms']} ms")
    print(f"  Max             : {metrics['latency_max_ms']} ms")
    print()
    print("  ── Model Inference Time (pure forward pass) ────────────")
    print(f"  P50             : {metrics['inference_p50_ms']} ms")
    print(f"  P95             : {metrics['inference_p95_ms']} ms")
    print(f"  P99             : {metrics['inference_p99_ms']} ms")
    print(f"  Mean            : {metrics['inference_mean_ms']} ms")
    print()
    print("  ── System Resources ────────────────────────────────────")
    print(f"  CPU avg         : {metrics['cpu_avg_pct']} %")
    print(f"  CPU max         : {metrics['cpu_max_pct']} %")
    print(f"  Memory avg      : {metrics['mem_avg_mb']} MB")
    print(f"  Memory max      : {metrics['mem_max_mb']} MB")
    print("=" * 60 + "\n")


async def warmup(url: str, n: int = 5) -> None:
    print(f"Warming up with {n} requests...")
    async with httpx.AsyncClient() as client:
        for i in range(n):
            await send_request(client, url, SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)])
    print("Warmup complete.\n")


async def check_health(url: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{url}/health", timeout=10.0)
            data = resp.json()
            return data.get("model_loaded", False)
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark rubert-mini-frida inference service")
    parser.add_argument("--url", default="http://localhost:8000", help="Service base URL")
    parser.add_argument("--requests", type=int, default=200, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup requests")
    parser.add_argument("--output-json", default=None, help="Optional path to save metrics as JSON")
    args = parser.parse_args()

    print(f"Checking service health at {args.url}...")
    healthy = await check_health(args.url)
    if not healthy:
        print("ERROR: Service is not healthy or model not loaded. Aborting.")
        sys.exit(1)
    print("Service is healthy.\n")

    if args.warmup > 0:
        await warmup(args.url, args.warmup)

    print(f"Starting load test: {args.requests} requests, concurrency={args.concurrency}")
    data = await run_load_test(args.url, args.requests, args.concurrency)
    metrics = compute_metrics(data)

    print_report(metrics, args.concurrency)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_json}")


if __name__ == "__main__":
    asyncio.run(main())
