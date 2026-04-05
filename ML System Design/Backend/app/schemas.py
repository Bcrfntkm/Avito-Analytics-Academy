from pydantic import BaseModel, Field
from typing import List


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Input text to embed")


class EmbedResponse(BaseModel):
    embedding: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Model name used for embedding")
    processing_time_ms: float = Field(..., description="Pure model inference time in milliseconds")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
