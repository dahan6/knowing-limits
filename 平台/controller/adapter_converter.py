"""Adapter converter: HF safetensors → GGUF for llama.cpp."""
import subprocess
from pathlib import Path
from typing import Optional

import config


class AdapterConverter:
    """Convert trained HF adapters to GGUF format for llama.cpp deployment."""

    def __init__(self, llama_cpp_path: str = None):
        self.llama_cpp_path = Path(
            llama_cpp_path or
            Path.home() / "ai-training/glm-4.5-air/llama.cpp/llama-src"
        )
        self.convert_script = self.llama_cpp_path / "convert_lora_to_gguf.py"

    def convert(
        self,
        hf_adapter_path: str,
        output_dir: str = None,
        base_model_path: str = None,
    ) -> str:
        """
        Convert HF adapter to GGUF.
        Returns the GGUF adapter path.
        """
        hf_adapter_path = Path(hf_adapter_path)
        if not hf_adapter_path.exists():
            raise FileNotFoundError(f"HF adapter not found: {hf_adapter_path}")

        output_dir = Path(output_dir or hf_adapter_path / "gguf")
        output_dir.mkdir(parents=True, exist_ok=True)

        gguf_path = output_dir / "adapter.gguf"

        # Build conversion command
        cmd = [
            "python3",
            str(self.convert_script),
            "--outfile", str(gguf_path),
        ]

        if base_model_path:
            cmd.extend(["--base", str(base_model_path)])

        # Run conversion
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.llama_cpp_path),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Adapter conversion failed: {result.stderr}"
            )

        return str(gguf_path)

    def merge_and_quantize(
        self,
        base_gguf: str,
        adapter_gguf: str,
        output_path: str,
        quantize_type: str = "Q5_K_M",
    ) -> str:
        """
        Merge adapter into base model and re-quantize.
        This is optional and requires llama.cpp's llama-quantize tool.
        """
        # Note: llama.cpp doesn't directly support merging adapters
        # This would require dequantizing, merging, and re-quantizing
        # which is complex and may lose quality.
        # For V0.1, we recommend using adapter files directly.
        raise NotImplementedError(
            "Merge and quantize is not recommended for V0.1. "
            "Use adapter files directly with llama-server --lora."
        )


def convert_adapter(
    hf_adapter_path: str,
    output_dir: str = None,
    base_model_path: str = None,
) -> str:
    """Quick adapter conversion."""
    converter = AdapterConverter()
    return converter.convert(hf_adapter_path, output_dir, base_model_path)
