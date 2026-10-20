# 本地 AI 知识库（Gradio + Chroma + BGE-M3 + GLM-4.5-Air）

一个轻量、可修改的本地知识库系统。

- **Linux 后端**：负责文件解析、Embedding、向量检索、调用 GLM-4.5-Air
- **Windows 前端**：浏览器打开页面，拖拽上传文件，进行知识库问答

## 目录

- `app.py`：Gradio Web 界面入口
- `config.py`：配置（模型地址、端口、路径等）
- `embeddings.py`：FastEmbed 封装
- `document_loader.py`：PDF/Word/文本解析
- `knowledge_base.py`：Chroma 向量库操作
- `llm.py`：调用本地 GLM-4.5-Air API
- `requirements.txt`：Python 依赖
- `start.sh`：启动脚本
- `knowledge-base.service`：systemd 用户服务配置

## 快速开始

```bash
cd ~/ai-training/glm-4.5-air/knowledge-base
source venv/bin/activate
python app.py
```

浏览器访问：

```
http://192.168.10.2:7860
```

## 使用方法

1. 进入 **📚 知识库管理** 标签页
2. 输入知识库名称，拖拽文件上传
3. 点击 **上传并建立索引**
4. 进入 **💬 知识库问答** 标签页
5. 选择知识库，输入问题

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `http://192.168.10.2:8080/v1` | 大模型 API 地址 |
| `LLM_API_KEY` | `sk-local` | API Key |
| `LLM_MODEL` | `glm-4.5-air` | 模型名称 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding 模型 |
| `KB_HOST` | `0.0.0.0` | 服务监听地址 |
| `KB_PORT` | `7860` | 服务端口 |

## 支持文件格式

`.txt`、`.md`、`.pdf`、`.docx`、`.csv`、`.json`、`.py`、`.js`、`.ts`、`.html`、`.xml`
