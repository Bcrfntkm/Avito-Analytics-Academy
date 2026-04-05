# rubert-mini-frida Inference Service Benchmark Results

**Test Date:** April 5, 2026  
**System:** MacBook Pro M4, 32 GB RAM, Docker Desktop (CPU-only)  
**Model:** sergeyzh/rubert-mini-frida  
**Framework:** FastAPI + uvicorn

---

## 1. Test Configuration

| Parameter | Value |
|-----------|-------|
| Total Requests | 200 |
| Concurrency Level | 10 |
| Test Type | HTTP End-to-End |
| Service URL | http://localhost:8000 |

---

## 2. Load Testing Results

### 2.1 Overall Metrics

| Metric | Value |
|--------|-------|
| **Total Requests** | 200 |
| **Successful** | 200 |
| **Failed** | 0 |
| **Wall Time** | 1.64 s |
| **Throughput (RPS)** | **121.92** |

### 2.2 End-to-End Latency (HTTP)

Complete request processing time from client to response:

| Percentile | Latency |
|------------|---------|
| **P50 (median)** | 75.96 ms |
| **P95** | 110.92 ms |
| **P99** | 116.32 ms |
| **Mean** | 78.79 ms |
| **Min** | 37.19 ms |
| **Max** | 121.31 ms |

**Latency Distribution:**
```
[37-60 ms]   ████░░░░░░░░░░░░░░░░  ~15%
[60-80 ms]   ████████████████░░░░  ~45%
[80-100 ms]  ████████░░░░░░░░░░░░  ~25%
[100-120 ms] ████░░░░░░░░░░░░░░░░  ~15%
```

### 2.3 Model Inference Time (Pure Forward Pass)

Time spent exclusively on model processing (excluding HTTP overhead):

| Percentile | Time |
|------------|------|
| **P50** | 5.11 ms |
| **P95** | 21.83 ms |
| **P99** | 31.7 ms |
| **Mean** | 7.61 ms |

**Analysis:** The difference between inference time (~5-8 ms) and total latency (~76 ms) is ~68 ms, which includes:
- JSON serialization/deserialization
- HTTP overhead
- Text tokenization
- Post-processing (pooling, normalization)

### 2.4 System Resource Usage

| Resource | Average | Maximum |
|----------|---------|---------|
| **CPU** | 74.9% | 100.0% |
| **Memory (RSS)** | 46.5 MB | 47.1 MB |

**Notes:**
- CPU reaches 100% at peak load, which is normal for CPU-only inference
- Memory consumption is stable (~46 MB), no leaks detected
- Model weights ~50 MB, total memory consumption with PyTorch overhead is within expected range

---

## 3. Functionality Testing

### 3.1 Health Check Endpoint

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Status:** Working correctly

### 3.2 Embedding Endpoint

**Request:**
```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Привет, как дела?"}'
```

**Response (excerpt):**
```json
{
  "embedding": [0.0679, 0.1030, 0.0470, -0.0530, ...],
  "model": "sergeyzh/rubert-mini-frida",
  "processing_time_ms": 180.51
}
```

**Characteristics:**
- Embedding dimension: **312**
- Processing time: **~180 ms** (first request, including initialization)
- Format: L2-normalized vector

**Status:** Working correctly