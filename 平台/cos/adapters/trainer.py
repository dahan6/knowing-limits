"""QLoRA trainer for GLM-4.5-Air implicit memory fine-tuning."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from ..config import COSConfig


class QLoRATrainer:
    """QLoRA fine-tuning trainer with 4-bit quantization and gradient checkpointing."""

    def __init__(self, model_path: str = None, output_dir: str = None):
        cfg = COSConfig.from_env()
        self.model_path = model_path or cfg.local.hf_model_path
        self.output_dir = Path(output_dir or cfg.local.qlora_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_dataset(self, samples: List[Dict]) -> Dataset:
        """Convert samples to HF Dataset."""
        texts = []
        for s in samples:
            instruction = s.get("instruction", "")
            output = s.get("output", "")
            # Format as instruction-following text
            text = f"用户：{instruction}\n\n助手：{output}"
            texts.append({"text": text})
        return Dataset.from_list(texts)

    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """4-bit NF4 quantization config."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    def _get_lora_config(self, rank: int, alpha: int) -> LoraConfig:
        """LoRA configuration targeting attention and top MoE experts."""
        return LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def train(
        self,
        samples: List[Dict],
        adapter_name: str,
        rank: int = None,
        alpha: int = None,
        epochs: int = None,
        batch_size: int = None,
        grad_accum: int = None,
        learning_rate: float = None,
    ) -> str:
        """
        Train a QLoRA adapter.
        Returns the adapter output directory.
        """
        cfg = COSConfig.from_env()
        rank = rank or cfg.local.qlora_rank
        alpha = alpha or cfg.local.qlora_alpha
        epochs = epochs or cfg.local.qlora_epochs
        batch_size = batch_size or cfg.local.qlora_batch_size
        grad_accum = grad_accum or cfg.local.qlora_grad_accum
        learning_rate = learning_rate or cfg.local.qlora_learning_rate

        adapter_dir = self.output_dir / adapter_name
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=self._get_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        lora_config = self._get_lora_config(rank, alpha)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Build dataset
        dataset = self._build_dataset(samples)

        def tokenize_fn(examples):
            outputs = tokenizer(
                examples["text"],
                truncation=True,
                max_length=2048,
                padding="max_length",
                return_tensors=None,
            )
            outputs["labels"] = outputs["input_ids"].copy()
            return outputs

        tokenized_dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=dataset.column_names,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(adapter_dir),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=5,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            report_to="none",
            max_grad_norm=0.3,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        # Train
        trainer.train()

        # Save adapter
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        # Save training metadata
        metadata = {
            "adapter_name": adapter_name,
            "base_model": self.model_path,
            "rank": rank,
            "alpha": alpha,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "learning_rate": learning_rate,
            "num_samples": len(samples),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        with open(adapter_dir / "training_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return str(adapter_dir)


# Convenience function for quick training
def train_adapter(
    samples: List[Dict],
    adapter_name: str,
    model_path: str = None,
    output_dir: str = None,
    **kwargs,
) -> str:
    """Quick adapter training."""
    trainer = QLoRATrainer(model_path=model_path, output_dir=output_dir)
    return trainer.train(samples, adapter_name, **kwargs)
