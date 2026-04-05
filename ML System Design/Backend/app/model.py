import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Tuple

MODEL_NAME = "sergeyzh/rubert-mini-frida"

class ModelHolder:
    """Singleton holder for the model and tokenizer."""
    tokenizer = None
    model = None
    is_loaded: bool = False

    @classmethod
    def load(cls) -> None:
        """Load model and tokenizer from HuggingFace Hub."""
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        cls.model = AutoModel.from_pretrained(MODEL_NAME)
        cls.model.eval()
        cls.is_loaded = True

    @classmethod
    def embed(cls, text: str) -> Tuple[list, float]:
        """
        Generate embedding for the given text.
        Returns (embedding_list, inference_time_ms).
        """
        if not cls.is_loaded:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()

        encoded = cls.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            output = cls.model(**encoded)

        # Mean pooling with attention mask
        attention_mask = encoded["attention_mask"]
        token_embeddings = output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

        # L2 normalization
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return embedding[0].tolist(), elapsed_ms
