# Controller V0.1 使用说明

## 1. 定位

Controller 是认知操作系统的“调度中心”。它接收用户问题后：

1. 结合**短期记忆**做**上下文思考**，改写/补全当前问题；
2. 对改写后的问题做**查询扩写**；
3. **并行**执行 **KB 检索** 与 **LLM 知识检测**；
4. 根据 KB 分数与 LLM known 概率的四种组合走不同分支；
5. 输出带标注的最终回答，并更新短期记忆。

## 2. 检索链路

### KB 检索

```
查询扩写
    │
    ├─► 向量 + 关键词 查询
    ├─► 重排序查询
    ├─► 查询扩展（同义/补全）
    └─► 上下文思考
            │
            ▼
    KB 向量检索（BGE-M3 + BM25 + rerank）
            │
            ▼
    KB 检索结果（score）
```

### LLM 检索

```
Hybrid + Expanded 查询生成
    │
    ▼
上下文思考
    │
    ▼
SVM-Linear 分类器
    │
    ▼
LLM 检测结果（known probability）
```

## 3. 四象限分类

| 情况 | KB 分数 | LLM known 概率 | 处理路径 | source |
|---|---|---|---|---|
| **A** | 高（≥0.6） | 高（≥0.7） | 直接 KB-RAG 生成，不改记忆 | `kb` |
| **B** | 高（≥0.6） | 低（<0.7） | 依 KB 做逻辑推理，多路影子验证，frozen LoRA 后训练 | `kb_reasoned` |
| **C** | 低（<0.6） | 高（≥0.7） | MoE 多路推理 + 网查对比，验证合理后写入 KB | `llm_moe` |
| **D** | 低（<0.6） | 低（<0.7） | 联网搜索摘要补充进 KB，然后进入 B 流程整理学习 | `unknown` |

## 4. 三种记忆

| 记忆类型 | 对应实现 | 说明 |
|---|---|---|
| 短期记忆 | `ContextMemory` | 最近 N 轮对话，用于上下文理解和指代消解 |
| 长期显式记忆 | KB（Chroma + BGE-M3 + BM25） | 可检索、可回写的知识库 |
| 长期隐式记忆 | LLM 内部权重 | 黑箱参数记忆，通过检测器评估 |

## 5. 目录结构

```
controller/
├── __init__.py
├── config.py              # 配置项
├── context_memory.py      # 短期记忆
├── query_expander.py      # 查询扩写
├── kb_client.py           # 知识库 HTTP 客户端（含写入）
├── web_search.py          # DuckDuckGo 搜索客户端
├── llm_client.py          # 本地 GLM 调用客户端
├── classifier.py          # LLM 知识检测器封装
├── controller.py          # 核心状态机
├── main.py                # CLI 入口（支持交互模式）
├── requirements.txt
├── architecture.drawio    # 架构图源文件
├── ARCHITECTURE.md        # 架构说明
└── CONTROLLER_V0.1.md     # 本文档
```

## 6. 前置依赖

1. **llama.cpp server** 在 `http://127.0.0.1:8080` 运行。
2. **知识库 API** 在 `http://127.0.0.1:7861` 运行（含 `/v1/search` 与 `/v1/add_snippets`）。
3. Python 环境 `ai` 中安装依赖：

```bash
cd ~/ai-training/glm-4.5-air/controller
conda run -n ai pip install -r requirements.txt
```

## 7. 运行方式

### 7.1 启动知识库 API

```bash
cd ~/ai-training/glm-4.5-air/knowledge-base
source venv/bin/activate
python api_server.py
```

### 7.2 启动 Controller CLI

单条问题：

```bash
cd ~/ai-training/glm-4.5-air/controller
conda run -n ai python3 main.py "中国的首都是哪里？"
```

交互模式（保留短期记忆）：

```bash
conda run -n ai python3 main.py
```

## 8. 分支示例

### A. KB 高 + LLM 高

```bash
python3 main.py "双机直连时 Linux 的 IP 是多少？"
```

`case: "A"`, `source: "kb"`。

### B. KB 高 + LLM 低

```bash
python3 main.py "Linux 主机的 MAC 地址是多少？"
```

`case: "B"`, `source: "kb_reasoned"`。

### C. KB 低 + LLM 高

```bash
python3 main.py "中国的首都是哪里？"
```

`case: "C"`, `source: "llm_moe"`。

### D. KB 低 + LLM 低

```bash
python3 main.py "2050年诺贝尔文学奖得主是谁？"
```

`case: "D"`, `source: "unknown"`。

## 9. 关键配置（`config.py`）

| 配置项 | 默认值 | 说明 |
|---|---|---|
| `KB_API_BASE` | `http://127.0.0.1:7861` | 知识库 API 地址 |
| `DEFAULT_KB_NAME` | `""` | 默认知识库名 |
| `LLM_BASE_URL` | `http://127.0.0.1:8080/v1` | 本地大模型 API |
| `KB_HIT_HIGH` | `0.6` | KB 高命中阈值 |
| `LLM_KNOWN_HIGH` | `0.7` | LLM 高置信阈值 |
| `ENABLE_WEB_SEARCH` | `True` | 是否启用联网搜索 |
| `QUERY_EXPANSION_N` | `2` | 查询扩写数量 |
| `MAX_CONTEXT_TURNS` | `5` | 短期记忆保留轮数 |

## 10. 输出字段说明

| 字段 | 类型 | 含义 |
|---|---|---|
| `case` | str | 四象限分类：`A/B/C/D` |
| `source` | str | 答案来源 |
| `contextualized_query` | str | 结合上下文改写后的查询 |
| `context` | str | 当前短期记忆文本 |
| `expanded_queries` | list | 扩写后的查询列表 |
| `kb_score` | float | KB 最高命中分数 |
| `kb_chunks` | list | KB 检索结果 |
| `llm_detection` | dict | LLM 知识检测完整结果 |
| `web_results` | list | 联网搜索摘要 |
| `moe_paths` | list | MoE 多路路径（仅 case C） |
| `verification` | dict | 结论验证结果（仅 case C） |
| `kb_added_chunks` | int | 回写到 KB 的 chunk 数量 |
| `answer` | str | 最终回答 |
| `confidence` | float | 置信度 |
| `needs_fact_check` | bool | 是否建议事实核查 |

## 11. 作为库调用

```python
from controller import Controller

ctrl = Controller()
result = ctrl.process("中国的首都是哪里？")
print(result["case"], result["source"], result["answer"])

# 多轮对话保留短期记忆
result2 = ctrl.process("那上海呢？")  # 会结合上文理解“那”指什么
```

## 12. 已知限制

- 阈值 `KB_HIT_HIGH=0.6`、`LLM_KNOWN_HIGH=0.7` 需根据数据校准；
- LLM 检测仍需要约 9 次 `/completion` 调用；
- 联网搜索稳定性受网络环境影响；
- 真正 frozen LoRA 后训练尚未接入，目前只记录学习日志。

## 13. 下一步建议

1. 跑一批真实问题，校准阈值；
2. 把 `verification` 从 JSON 解析改成更稳定的评分模型；
3. 接入真正的影子验证与 frozen LoRA 微调管线；
4. 把 Controller 封装成 FastAPI 服务。
