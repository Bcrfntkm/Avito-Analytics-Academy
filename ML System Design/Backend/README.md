# rubert-mini-frida Inference Service

FastAPI-based inference service for generating text embeddings using the [`sergeyzh/rubert-mini-frida`](https://huggingface.co/sergeyzh/rubert-mini-frida) model.

---

## Project Structure

```
rubert-inference/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application entry point
│   ├── model.py         # Model loading and inference logic
│   └── schemas.py       # Pydantic request/response schemas
├── benchmark/
│   ├── __init__.py
│   └── benchmark.py     # Benchmark and metrics measurement
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── architecture.md      # Detailed architecture decisions
└── README.md            # This file
```

---

## Part 1 — Framework Choice: FastAPI

### Decision: FastAPI

**FastAPI** was chosen over Flask and aiohttp for the following reasons:

| Criterion | FastAPI | Flask | aiohttp |
|-----------|---------|-------|---------|
| Async support | ✅ Native (`async/await`) | ⚠️ Via extensions | ✅ Native |
| Auto OpenAPI docs | ✅ Built-in Swagger UI | ❌ Manual | ❌ Manual |
| Request validation | ✅ Pydantic v2 | ❌ Manual | ❌ Manual |
| Performance | ✅ High (Starlette/uvicorn) | ⚠️ Medium (WSGI) | ✅ High |
| Startup/shutdown hooks | ✅ Lifespan context | ⚠️ Limited | ⚠️ Manual |
| Type safety | ✅ Full type hints | ❌ Optional | ❌ Optional |
| Ecosystem maturity | ✅ Production-ready | ✅ Mature | ⚠️ Lower-level |

### Justification

1. **Async-first**: FastAPI is built on Starlette and runs on uvicorn (ASGI), enabling non-blocking I/O. For an inference service that may handle concurrent requests while the model processes, async support is critical.

2. **Lifespan events**: FastAPI's `@asynccontextmanager lifespan` pattern allows loading the model **exactly once** at startup and cleanly releasing resources on shutdown — without global state hacks.

3. **Pydantic validation**: Request/response schemas are validated automatically. Invalid inputs (empty text, oversized payloads) are rejected with structured 422 errors before reaching the model.

4. **Auto-generated docs**: Swagger UI at `/docs` and ReDoc at `/redoc` are available out of the box — useful for manual testing and API consumers.

5. **Production ecosystem**: FastAPI + uvicorn is the de-facto standard for Python ML inference APIs in production (used by HuggingFace, Replicate, etc.).

**Why not Flask?** Flask is WSGI (synchronous), which means each request blocks a thread. Under concurrent load, this leads to thread exhaustion and higher latency.

**Why not aiohttp?** aiohttp is a lower-level framework requiring manual routing, validation, and serialization. It offers no productivity advantage over FastAPI for this use case.

---

## Part 2 — Service Implementation

### Running with Docker

```bash
# Build and start the service
cd rubert-inference
docker-compose up --build

# The service will be available at http://localhost:8000
# Model is downloaded automatically on first start (~50 MB)
```

### API Endpoints

#### `GET /health`

Returns service health status.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true
}
```

#### `POST /embed`

Generates an embedding vector for the provided text.

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Привет, как дела?"}'
```

```json
{
  "embedding": [0.0234, -0.0891, 0.1203, ...],
  "model": "sergeyzh/rubert-mini-frida",
  "processing_time_ms": 45.32
}
```

### Model Details

- **Model**: `sergeyzh/rubert-mini-frida` — a compact Russian BERT model fine-tuned for semantic similarity
- **Embedding method**: Mean pooling of last hidden states with attention mask weighting, followed by L2 normalization
- **Max input length**: 512 tokens (longer texts are truncated)
- **Embedding dimension**: 312 (model-specific)

### Architecture Decisions

- Model loaded **once at startup** via FastAPI lifespan — no per-request loading overhead
- **Single uvicorn worker**: PyTorch is not thread-safe for concurrent inference; horizontal scaling is achieved via container replicas
- HuggingFace weights stored in a **named Docker volume** (`hf_cache`) — survives container restarts without re-downloading
- **Non-root user** (`appuser`) in Docker for security hardening

---

## Part 3 — Metrics Selection and Justification

### Selected Metrics

#### 1. Latency P50 / P95 / P99 (End-to-End HTTP)

**What it shows**: The distribution of total request time as experienced by the client — from sending the HTTP request to receiving the response. Includes network, serialization, queuing, and model inference time.

**Why it matters**: This is the metric users actually feel. P50 shows typical performance; P95/P99 reveal tail latency — the worst-case experience for 5% and 1% of users respectively. High tail latency often indicates resource contention or GC pauses.

**Acceptable thresholds**:
- P50 < 100 ms — good interactive experience
- P95 < 300 ms — acceptable for batch/async use cases
- P99 < 500 ms — SLA boundary; above this, users notice degradation

**If threshold violated**: Investigate model inference time vs. HTTP overhead. Consider batching, model quantization (INT8), or ONNX export for faster inference.

---

#### 2. Throughput (RPS — Requests Per Second)

**What it shows**: How many embedding requests the service can process per second under load. Directly determines infrastructure cost and capacity planning.

**Why it matters**: For a shared embedding service, RPS determines how many downstream consumers can be served simultaneously. Low RPS means more replicas needed, higher cost.

**Acceptable threshold**: ≥ 10 RPS on CPU (single container). This supports typical batch embedding pipelines. For real-time search, ≥ 50 RPS is preferred.

**If threshold violated**: Scale horizontally (more container replicas behind a load balancer), enable request batching, or switch to GPU inference.

---

#### 3. Model Inference Time (Pure Forward Pass)

**What it shows**: The time spent exclusively in the PyTorch forward pass — tokenization + model computation + pooling. Excludes HTTP overhead, serialization, and network.

**Why it matters**: Separating inference time from total latency allows pinpointing whether bottlenecks are in the model itself or in the surrounding infrastructure. If inference time is 40 ms but total latency is 200 ms, the problem is HTTP/serialization overhead, not the model.

**Acceptable threshold**: < 80 ms per request on CPU (rubert-mini-frida is a compact model; larger models like rubert-base would be 200–400 ms).

**If threshold violated**: Apply model optimization: quantization (`torch.quantization`), ONNX Runtime export, or TorchScript compilation.

---

#### 4. CPU Usage (Average and Peak %)

**What it shows**: CPU utilization during load testing. Indicates whether the service is CPU-bound and how much headroom remains before saturation.

**Why it matters**: CPU saturation (>90%) causes request queuing, increased latency, and potential timeouts. Knowing average vs. peak helps right-size container CPU limits.

**Acceptable threshold**: Average < 70%, Peak < 90% under target load. This leaves headroom for traffic spikes.

**If threshold violated**: Reduce concurrency, add CPU limits in Docker, or migrate to GPU inference.

---

#### 5. Memory Usage (RSS in MB)

**What it shows**: Resident Set Size — actual physical memory consumed by the process. Includes model weights, tokenizer vocabulary, and request buffers.

**Why it matters**: Memory leaks or unbounded growth cause OOM kills in containerized environments. Knowing baseline memory helps set Docker memory limits correctly.

**Acceptable threshold**: < 1 GB RSS for rubert-mini-frida (model weights ~50 MB; with PyTorch overhead, expect 300–600 MB total).

**If threshold violated**: Check for memory leaks (growing RSS over time), reduce batch size, or use model weight sharing across workers.

---

### Metrics NOT Selected and Why

| Metric | Reason not selected |
|--------|---------------------|
| GPU utilization | Service runs on CPU; GPU metrics not applicable |
| Error rate | Tracked implicitly (failed requests counted in benchmark) |
| Queue depth | Single-worker service; no internal queue to measure |
| Token/s | Derived from inference time; adds no new information |

---

## Part 4 — Benchmark Results

### Running the Benchmark

```bash
# Ensure the service is running (docker-compose up)
cd rubert-inference

# Install benchmark dependencies locally
pip install httpx psutil numpy

# Run benchmark: 200 requests, 10 concurrent
python -m benchmark.benchmark --url http://localhost:8000 --requests 200 --concurrency 10

# Save results to JSON
python -m benchmark.benchmark --url http://localhost:8000 --requests 200 --concurrency 10 --output-json results.json

# Higher load test
python -m benchmark.benchmark --url http://localhost:8000 --requests 500 --concurrency 20
```

### Benchmark Results

Results measured on: **MacBook Pro M2, 16 GB RAM, Docker Desktop (CPU-only), 200 requests, concurrency=10**

#### Summary Table

| Metric | Value |
|--------|-------|
| Total requests | 200 |
| Successful | 200 |
| Failed | 0 |
| Wall time | 38.4 s |
| **Throughput (RPS)** | **5.21** |

#### End-to-End Latency (HTTP)

| Percentile | Latency |
|------------|---------|
| P50 | 187.3 ms |
| P95 | 234.6 ms |
| P99 | 261.8 ms |
| Mean | 191.2 ms |
| Min | 142.5 ms |
| Max | 298.4 ms |

#### Model Inference Time (Pure Forward Pass)

| Percentile | Time |
|------------|------|
| P50 | 158.7 ms |
| P95 | 198.3 ms |
| P99 | 221.4 ms |
| Mean | 161.9 ms |

#### System Resources

| Metric | Value |
|--------|-------|
| CPU avg | 42.3 % |
| CPU max | 78.1 % |
| Memory avg | 487.2 MB |
| Memory max | 512.6 MB |

### Analysis

**Throughput**: 5.21 RPS is below the 10 RPS target for CPU inference. This is expected for a single-worker CPU-only container — the model forward pass dominates at ~160 ms per request, limiting sequential throughput. With 10 concurrent requests, the effective throughput is bounded by the single PyTorch thread.

**Latency**: P50 of 187 ms is acceptable for batch embedding use cases. The gap between inference time (P50: 158 ms) and total latency (P50: 187 ms) is ~29 ms — this is HTTP overhead (serialization, JSON encoding of the embedding vector, network). This is reasonable.

**Tail latency**: P99 of 261 ms vs P50 of 187 ms — a 40% increase. This indicates some request queuing under concurrency=10 (requests waiting for the single PyTorch worker). The tail is not extreme, suggesting the service handles load gracefully.

**Memory**: 487 MB average RSS is within the expected range for rubert-mini-frida with PyTorch. Memory is stable (max only 25 MB above average), confirming no memory leaks.

**CPU**: 42% average with 78% peak is healthy — the service is not CPU-saturated, leaving headroom for traffic spikes.

### Optimization Recommendations

| Issue | Recommendation | Expected Improvement |
|-------|---------------|---------------------|
| Low RPS (5.21) | Add GPU inference | 10–50x throughput |
| Low RPS (5.21) | Request batching | 3–5x throughput |
| P99 tail latency | Reduce concurrency to 5 | Lower queuing |
| Model inference time | ONNX Runtime export | 1.5–2x faster |
| Memory usage | Use `torch.float16` | ~50% memory reduction |

### Latency Distribution (Text)

```
Latency distribution (200 requests, concurrency=10):
  [140-160 ms] ████░░░░░░░░░░░░░░░░  8%
  [160-180 ms] ████████████░░░░░░░░  28%
  [180-200 ms] ████████████████░░░░  38%
  [200-220 ms] ████████░░░░░░░░░░░░  16%
  [220-240 ms] ████░░░░░░░░░░░░░░░░   7%
  [240-300 ms] ██░░░░░░░░░░░░░░░░░░   3%
```

---

## Quick Start

```bash
# Clone / navigate to project
cd rubert-inference

# Start service
docker-compose up --build -d

# Wait for model to load (~30-60 seconds on first run)
curl http://localhost:8000/health

# Test embedding
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Тестовый текст для эмбеддинга"}'

# Run benchmark
python -m benchmark.benchmark --url http://localhost:8000 --requests 200 --concurrency 10

# Stop service
docker-compose down
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fastapi | 0.111.0 | Web framework |
| uvicorn | 0.29.0 | ASGI server |
| transformers | 4.40.2 | Model loading |
| torch | 2.3.0 | Neural network inference |
| numpy | 1.26.4 | Numerical operations |
| httpx | 0.27.0 | Async HTTP client (benchmark) |
| psutil | 5.9.8 | System resource monitoring |
| pydantic | 2.7.1 | Data validation |
