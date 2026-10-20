"""ONNX Cross-Encoder reranker wrapper (BGE-reranker-v2-m3)."""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

import config


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Reranker:
    """ONNX reranker using BGE-reranker-v2-m3."""

    def __init__(
        self,
        model_dir: str = config.RERANKER_DIR,
        onnx_path: str = config.RERANKER_ONNX,
        tokenizer_dir: str = config.RERANKER_TOKENIZER_DIR,
    ):
        self.model_dir = Path(model_dir)
        self.onnx_path = Path(onnx_path)
        self.tokenizer_dir = Path(tokenizer_dir)
        if not self.onnx_path.exists():
            # Fallback: look for any onnx file in the model dir
            candidates = list(self.model_dir.rglob("*.onnx"))
            if candidates:
                self.onnx_path = candidates[0]
            else:
                raise FileNotFoundError(f"Reranker ONNX model not found: {onnx_path}")

        tokenizer_path = self.tokenizer_dir if self.tokenizer_dir.exists() else self.model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.max_length = 512  # safe default for the quantized model

    def _run_batch(self, queries: List[str], passages: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        inputs_onnx = {k: v.astype(np.int64) for k, v in inputs.items() if k in self.input_names}
        outputs = self.session.run(None, inputs_onnx)
        logits = outputs[0]
        if logits.ndim == 2:
            logits = logits[:, 0]
        return _sigmoid(logits)

    def rerank(
        self,
        query: str,
        passages: List[Tuple[str, str, dict]],
        batch_size: int = 8,
    ) -> List[Tuple[str, str, dict, float]]:
        """
        Rerank (id, text, metadata) tuples for a query.
        Returns list sorted by descending relevance score.
        """
        scores = []
        for i in range(0, len(passages), batch_size):
            batch = passages[i : i + batch_size]
            queries = [query] * len(batch)
            texts = [p[1] for p in batch]
            batch_scores = self._run_batch(queries, texts)
            scores.extend(batch_scores.tolist())

        ranked = sorted(
            [(pid, text, meta, float(score)) for (pid, text, meta), score in zip(passages, scores)],
            key=lambda x: x[3],
            reverse=True,
        )
        return ranked
