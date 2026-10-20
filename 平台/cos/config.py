"""COS Framework 统一配置"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LocalConfig:
    """本地完整模式配置"""
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str = "glm-4.5-air"
    api_key: str = "sk-local"
    hf_model_path: str = ""  # QLoRA 训练用 HF 模型路径
    enable_qlora: bool = True
    qlora_output_dir: str = str(Path.home() / "ai-training/glm-4.5-air/adapters")
    qlora_rank: int = 8
    qlora_alpha: int = 16
    qlora_epochs: int = 2
    qlora_batch_size: int = 1
    qlora_grad_accum: int = 4
    qlora_learning_rate: float = 2e-4
    qlora_min_samples: int = 50
    qlora_shadow_test_size: int = 100
    qlora_max_degradation: float = 0.05

    @classmethod
    def from_env(cls) -> "LocalConfig":
        return cls(
            base_url=os.getenv("LOCAL_LLM_BASE_URL", cls.base_url),
            model=os.getenv("LOCAL_LLM_MODEL", cls.model),
            api_key=os.getenv("LOCAL_LLM_API_KEY", cls.api_key),
            hf_model_path=os.getenv("LOCAL_HF_MODEL_PATH", cls.hf_model_path),
            enable_qlora=os.getenv("LOCAL_ENABLE_QLORA", "true").lower() in ("1", "true", "yes"),
            qlora_rank=int(os.getenv("QLORA_RANK", str(cls.qlora_rank))),
            qlora_alpha=int(os.getenv("QLORA_ALPHA", str(cls.qlora_alpha))),
            qlora_epochs=int(os.getenv("QLORA_EPOCHS", str(cls.qlora_epochs))),
            qlora_batch_size=int(os.getenv("QLORA_BATCH_SIZE", str(cls.qlora_batch_size))),
            qlora_grad_accum=int(os.getenv("QLORA_GRAD_ACCUM", str(cls.qlora_grad_accum))),
            qlora_learning_rate=float(os.getenv("QLORA_LEARNING_RATE", str(cls.qlora_learning_rate))),
            qlora_min_samples=int(os.getenv("QLORA_MIN_SAMPLES", str(cls.qlora_min_samples))),
            qlora_shadow_test_size=int(os.getenv("QLORA_SHADOW_TEST_SIZE", str(cls.qlora_shadow_test_size))),
            qlora_max_degradation=float(os.getenv("QLORA_MAX_DEGRADATION", str(cls.qlora_max_degradation))),
        )


@dataclass
class RemoteConfig:
    """远程轻量模式配置"""
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4"
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.6
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "RemoteConfig":
        return cls(
            base_url=os.getenv("REMOTE_BASE_URL", cls.base_url),
            api_key=os.getenv("REMOTE_API_KEY", cls.api_key),
            model=os.getenv("REMOTE_MODEL", cls.model),
            timeout=int(os.getenv("REMOTE_TIMEOUT", str(cls.timeout))),
            max_retries=int(os.getenv("REMOTE_MAX_RETRIES", str(cls.max_retries))),
            temperature=float(os.getenv("REMOTE_TEMPERATURE", str(cls.temperature))),
            max_tokens=int(os.getenv("REMOTE_MAX_TOKENS", str(cls.max_tokens))),
        )


@dataclass
class COSConfig:
    """COS 框架统一配置"""
    mode: str = "local"  # local / remote
    local: LocalConfig = field(default_factory=LocalConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)

    # 通用配置（两种模式共享）
    kb_api_base: str = "http://127.0.0.1:7861"
    default_kb_name: str = ""
    kb_hit_high: float = 0.6
    kb_hit_low: float = 0.3
    kb_top_k: int = 5
    llm_known_high: float = 0.7
    llm_known_low: float = 0.4
    routing_borderline_kb: float = 0.15
    routing_borderline_llm: float = 0.15
    enable_web_search: bool = True
    web_search_top_k: int = 3
    web_search_region: str = "zh-cn"
    max_context_turns: int = 5
    enable_memory_value_estimation: bool = True
    dedup_similarity_threshold: float = 0.92
    memory_default_weight: float = 1.0
    memory_time_decay_lambda: float = 0.05
    memory_access_weight: float = 0.3
    memory_confidence_weight: float = 0.5
    memory_archive_days: int = 90
    memory_topic_shift_threshold: float = 0.6
    llm_detector_model: str = str(Path.home() / "ai-training/glm-4.5-air/experiments/final_hybrid_detector_model.pkl")

    @classmethod
    def from_env(cls) -> "COSConfig":
        mode = os.getenv("COS_MODE", "local")
        config = cls(mode=mode)
        config.local = LocalConfig.from_env()
        config.remote = RemoteConfig.from_env()

        # 通用配置
        config.kb_api_base = os.getenv("KB_API_BASE", config.kb_api_base)
        config.default_kb_name = os.getenv("DEFAULT_KB_NAME", config.default_kb_name)
        config.kb_hit_high = float(os.getenv("KB_HIT_HIGH", str(config.kb_hit_high)))
        config.kb_hit_low = float(os.getenv("KB_HIT_LOW", str(config.kb_hit_low)))
        config.kb_top_k = int(os.getenv("KB_TOP_K", str(config.kb_top_k)))
        config.llm_known_high = float(os.getenv("LLM_KNOWN_HIGH", str(config.llm_known_high)))
        config.llm_known_low = float(os.getenv("LLM_KNOWN_LOW", str(config.llm_known_low)))
        config.routing_borderline_kb = float(os.getenv("ROUTING_BORDERLINE_KB", str(config.routing_borderline_kb)))
        config.routing_borderline_llm = float(os.getenv("ROUTING_BORDERLINE_LLM", str(config.routing_borderline_llm)))
        config.enable_web_search = os.getenv("ENABLE_WEB_SEARCH", "true").lower() in ("1", "true", "yes")
        config.web_search_top_k = int(os.getenv("WEB_SEARCH_TOP_K", str(config.web_search_top_k)))
        config.web_search_region = os.getenv("WEB_SEARCH_REGION", config.web_search_region)
        config.max_context_turns = int(os.getenv("MAX_CONTEXT_TURNS", str(config.max_context_turns)))
        config.enable_memory_value_estimation = os.getenv("ENABLE_MEMORY_VALUE_ESTIMATION", "true").lower() in ("1", "true", "yes")

        return config
