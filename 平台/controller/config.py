"""Controller V0.1 配置"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# 外部服务地址
KB_API_BASE = os.getenv("KB_API_BASE", "http://127.0.0.1:7861")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4.5-air")
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-local")

# 知识库
DEFAULT_KB_NAME = os.getenv("DEFAULT_KB_NAME", "")
KB_HIT_HIGH = float(os.getenv("KB_HIT_HIGH", "0.6"))     # 强命中
KB_HIT_LOW = float(os.getenv("KB_HIT_LOW", "0.3"))       # 弱命中
KB_TOP_K = int(os.getenv("KB_TOP_K", "5"))

# LLM 知识检测
LLM_KNOWN_HIGH = float(os.getenv("LLM_KNOWN_HIGH", "0.7"))
LLM_KNOWN_LOW = float(os.getenv("LLM_KNOWN_LOW", "0.4"))

# 路由边界区：分数落在阈值附近时，启用多证据融合/校验以提高鲁棒性
ROUTING_BORDERLINE_KB = float(os.getenv("ROUTING_BORDERLINE_KB", "0.15"))
ROUTING_BORDERLINE_LLM = float(os.getenv("ROUTING_BORDERLINE_LLM", "0.15"))

# 查询扩写
QUERY_EXPANSION_N = int(os.getenv("QUERY_EXPANSION_N", "2"))

# 短期记忆
MAX_CONTEXT_TURNS = int(os.getenv("MAX_CONTEXT_TURNS", "5"))
LLM_DETECTOR_MODEL = Path(os.getenv(
    "LLM_DETECTOR_MODEL",
    str(Path.home() / "ai-training" / "glm-4.5-air" / "experiments" / "final_hybrid_detector_model.pkl"),
))

# 长期记忆（KB）策略
KB_ISOLATION_MODE = os.getenv("KB_ISOLATION_MODE", "single")  # single / by_topic / by_user
ENABLE_MEMORY_VALUE_ESTIMATION = os.getenv("ENABLE_MEMORY_VALUE_ESTIMATION", "true").lower() in ("1", "true", "yes")
DEDUP_SIMILARITY_THRESHOLD = float(os.getenv("DEDUP_SIMILARITY_THRESHOLD", "0.92"))
MEMORY_DEFAULT_WEIGHT = float(os.getenv("MEMORY_DEFAULT_WEIGHT", "1.0"))
MEMORY_TIME_DECAY_LAMBDA = float(os.getenv("MEMORY_TIME_DECAY_LAMBDA", "0.05"))  # 时间衰减系数
MEMORY_ACCESS_WEIGHT = float(os.getenv("MEMORY_ACCESS_WEIGHT", "0.3"))           # 频率提升系数
MEMORY_CONFIDENCE_WEIGHT = float(os.getenv("MEMORY_CONFIDENCE_WEIGHT", "0.5"))   # 置信度加成系数
MEMORY_ARCHIVE_DAYS = int(os.getenv("MEMORY_ARCHIVE_DAYS", "90"))                # 多久未访问视为冷记忆
MEMORY_TOPIC_SHIFT_THRESHOLD = float(os.getenv("MEMORY_TOPIC_SHIFT_THRESHOLD", "0.6"))  # 主题切换判断阈值

# 联网搜索
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() in ("1", "true", "yes")
WEB_SEARCH_TOP_K = int(os.getenv("WEB_SEARCH_TOP_K", "3"))
WEB_SEARCH_REGION = os.getenv("WEB_SEARCH_REGION", "zh-cn")

# 开放/主观问题关键词（命中则视为 open，不再尝试搜索）
OPEN_KEYWORDS = [
    "意义", "看法", "评价", "观点", "态度", "如何", "怎样", "为什么",
    "是否应该", "如何看待", "怎么", "怎样看待", "建议", "推荐", "创意",
    "写一", "描述", "想象", "假如", "如果", "假如", "感受",
]

# LLM 生成参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.6"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# QLoRA 训练配置
QLORA_MODEL_PATH = os.getenv("QLORA_MODEL_PATH", str(Path.home() / "ai-training/glm-4.5-air/models/glm-4.5-air-hf"))
QLORA_OUTPUT_DIR = os.getenv("QLORA_OUTPUT_DIR", str(Path.home() / "ai-training/glm-4.5-air/adapters"))
QLORA_RANK = int(os.getenv("QLORA_RANK", "8"))
QLORA_ALPHA = int(os.getenv("QLORA_ALPHA", "16"))
QLORA_EPOCHS = int(os.getenv("QLORA_EPOCHS", "2"))
QLORA_BATCH_SIZE = int(os.getenv("QLORA_BATCH_SIZE", "1"))
QLORA_GRAD_ACCUM = int(os.getenv("QLORA_GRAD_ACCUM", "4"))
QLORA_LEARNING_RATE = float(os.getenv("QLORA_LEARNING_RATE", "2e-4"))
QLORA_MIN_SAMPLES = int(os.getenv("QLORA_MIN_SAMPLES", "50"))
QLORA_SHADOW_TEST_SIZE = int(os.getenv("QLORA_SHADOW_TEST_SIZE", "100"))
QLORA_MAX_DEGRADATION = float(os.getenv("QLORA_MAX_DEGRADATION", "0.05"))
