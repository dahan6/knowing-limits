# COS Framework: 深盆科技认知操作系统框架

## 简介

COS Framework 是一个可复用的认知操作系统框架，支持两种部署模式：

- **本地完整模式**：本地 LLM + 三层记忆（短期上下文 + 显式 KB + 隐式 LoRA）
- **远程轻量模式**：外部 API LLM + 两层记忆（短期上下文 + 显式 KB）

## 快速开始

### 本地完整模式

```python
from cos import Controller, LLMProviderFactory, COSConfig

config = COSConfig.from_env()
config.mode = "local"
provider = LLMProviderFactory.create(config)
controller = Controller(llm_provider=provider, config=config)

result = controller.process("中国的首都是哪里？")
print(result["answer"])
```

### 远程轻量模式

```python
from cos import Controller, LLMProviderFactory, COSConfig

config = COSConfig.from_env()
config.mode = "remote"
provider = LLMProviderFactory.create(config)
controller = Controller(llm_provider=provider, config=config)

result = controller.process("中国的首都是哪里？")
print(result["answer"])
```

## 配置

通过环境变量配置：

```bash
# 本地模式
export COS_MODE=local
export LOCAL_LLM_BASE_URL=http://127.0.0.1:8080/v1
export LOCAL_HF_MODEL_PATH=~/ai-training/glm-4.5-air/models/glm-4.5-air-hf

# 远程模式
export COS_MODE=remote
export REMOTE_BASE_URL=https://api.openai.com/v1
export REMOTE_API_KEY=sk-xxx
export REMOTE_MODEL=gpt-4

# 通用配置
export KB_API_BASE=http://127.0.0.1:7861
export KB_HIT_HIGH=0.6
export LLM_KNOWN_HIGH=0.7
```

## 架构

```
┌─────────────────────────────────────────┐
│           Application Layer             │
│   CLI / API / WebUI / Third-party       │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│         COS Controller (通用逻辑)        │
│  输入 → 扩写 → 检索 → 路由 → 生成 → 记忆 │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Local Provider │       │ Remote Provider│
│  本地完整模式   │       │  远程轻量模式   │
│  - GGUF 推理   │       │  - OpenAI API │
│  - LoRA 微调   │       │  - Claude API │
│  - 三层记忆    │       │  - Kimi API   │
└───────────────┘       └───────────────┘
```

## 能力矩阵

| 功能 | 本地模式 | 远程模式 |
|------|---------|---------|
| 查询扩写 | ✅ | ✅ |
| KB 检索 | ✅ | ✅ |
| 四象限路由 | ✅ | ✅ |
| 融合生成 | ✅ | ✅ |
| 短期记忆 | ✅ | ✅ |
| 显式 KB | ✅ | ✅ |
| LoRA 微调 | ✅ | ❌ |
| Adapter 热插拔 | ✅ | ❌ |
| 影子验证 | ✅ | ❌ |

## 目录结构

```
cos/
├── config.py              # 统一配置
├── controller.py          # 核心控制器
├── providers/             # LLM Provider
│   ├── base.py            # 接口定义
│   ├── local.py           # 本地模式
│   └── remote.py          # 远程模式
├── memory/                # 记忆层
│   ├── context_memory.py  # 短期记忆
│   ├── kb_client.py       # KB 客户端
│   └── training_manager.py# 训练样本管理
├── routing/               # 路由层
│   ├── classifier.py      # LLM 知识检测
│   ├── router.py          # 四象限路由
│   └── fusion.py          # 融合生成
├── memory_value/          # 记忆价值评估
│   └── estimator.py
├── adapters/              # Adapter 管理
│   ├── registry.py        # 注册表
│   ├── trainer.py         # QLoRA 训练器
│   └── converter.py       # HF→GGUF 转换
├── utils/                 # 工具
│   ├── web_search.py      # 联网搜索
│   └── query_expander.py  # 查询扩写
├── examples/              # 示例
└── tests/                 # 测试
```

## 许可证

内部技术讨论文档，深盆科技（Synbasin）
