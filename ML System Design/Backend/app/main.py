from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from app.schemas import EmbedRequest, EmbedResponse, HealthResponse
from app.model import ModelHolder, MODEL_NAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    ModelHolder.load()
    yield
    # Cleanup if needed
    ModelHolder.is_loaded = False


app = FastAPI(
    title="rubert-mini-frida Inference Service",
    description="FastAPI service for generating text embeddings using rubert-mini-frida",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=ModelHolder.is_loaded,
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Generate embedding for the provided text."""
    if not ModelHolder.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        embedding, inference_time_ms = ModelHolder.embed(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return EmbedResponse(
        embedding=embedding,
        model=MODEL_NAME,
        processing_time_ms=inference_time_ms,
    )
