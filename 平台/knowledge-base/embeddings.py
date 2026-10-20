"""Embedding model wrapper using a local ONNX BGE-M3 model."""
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

import config


class EmbeddingFunction:
    """Chroma-compatible embedding function backed by ONNX Runtime."""

    def __init__(self, model_dir: str = config.EMBEDDING_MODEL):
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Embedding model directory not found: {model_dir}")

        onnx_path = self.model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        # Use CPU execution provider. The model is small enough for CPU.
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.max_length = 512  # BGE-M3 supports up to 8192; 512 is a good default.

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def _embed(self, texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        inputs_onnx = {k: v.astype(np.int64) for k, v in inputs.items()}
        outputs = self.session.run(None, inputs_onnx)
        # outputs order: dense (normalized by export), sparse, colbert
        dense = outputs[0]
        # Some export versions may not normalize; ensure unit vectors.
        return self._normalize(dense)

    def __call__(self, input: List[str]) -> List[List[float]]:
        if not input:
            return []
        embeddings = self._embed(input)
        return embeddings.tolist()

    def embed_query(self, input):
        texts = input if isinstance(input, list) else [input]
        return self(texts)

    def name(self) -> str:
        return "onnx_bge_m3"


# Singleton instance for the app
embedding_fn = EmbeddingFunction()
