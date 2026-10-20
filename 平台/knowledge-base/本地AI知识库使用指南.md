# 本地 AI 知识库使用指南（增强版）

## 1. 这是什么？

这是一个部署在 **Linux 工作站** 上的本地 RAG（Retrieval-Augmented Generation，检索增强生成）知识库：

- **前端页面**：基于 Gradio 的 Web UI
- **Embedding 模型**：本地 ONNX 版 BGE-M3（CPU 推理）
- **向量数据库**：Chroma（持久化存储在 `data/chroma`）
- **检索增强**：向量检索 + BM25 关键词检索 + RRF 融合 + BGE Reranker 重排序
- **查询理解**：查询改写/扩展、简单意图识别
- **上下文处理**：去重、压缩、带编号引用
- **多模态支持**：PDF/Word 表格提取、图片 OCR、Excel 读取
- **大模型**：本地部署的 GLM-4.5-Air（`192.168.10.2:8080/v1`）
- **访问方式**：在 Windows 主机上打开浏览器即可使用

工作流程：

```
用户上传文件 → 解析文本/表格/图片 → 切分文本块 → BGE-M3 向量化 → 存入 Chroma
用户提问 → 查询改写(可选) → 向量+关键词混合检索 → RRF 融合 → Reranker 重排序 →
上下文去重/压缩 → 拼接引用 → GLM-4.5-Air 生成带来源标注的回答
```

## 2. 如何从 Windows 访问？

确保 Linux 工作站已开机并联网，然后在 Windows 浏览器地址栏输入：

```
http://192.168.10.2:7860
```

即可看到知识库页面。

## 3. 页面功能说明

页面分为三个标签页：

### 3.1 📚 知识库管理

- **知识库名称**：输入一个名称，例如 `技术文档`、`论文集`、`项目资料`
- **分块大小 / 重叠**：控制文档切分的粒度，默认 500 字符、重叠 50 字符
- **拖拽文件到此处上传**：支持多文件同时上传
- **上传并建立索引**：把文件解析、切分、向量化后存入知识库
- **已有知识库**：选择知识库后可以看到其中包含的文件，并支持删除整个知识库

**支持的文件格式**：

| 格式 | 说明 |
|------|------|
| `.txt` / `.md` | 纯文本 / Markdown |
| `.pdf` | PDF 文档（自动提取文字、表格、内嵌图片并 OCR） |
| `.docx` | Word 文档（段落 + 表格） |
| `.csv` / `.xlsx` | 表格（自动转为 Markdown） |
| `.json` / `.py` / `.js` / `.ts` / `.html` / `.xml` | 代码与配置文件 |
| `.jpg` / `.jpeg` / `.png` / `.webp` / `.bmp` | 图片（自动 OCR 文字） |

### 3.2 💬 知识库问答

- **选择要使用的知识库**：从下拉框选择已创建的知识库
- **检索片段数量**：设置每次检索返回多少个相关文本块（默认 5）
- **混合检索（向量+关键词）**：同时启用语义检索和关键词匹配，推荐开启
- **重排序**：使用 BGE Reranker 对候选片段二次排序，推荐开启
- **查询扩展**：让大模型生成 2-3 个同义查询变体以扩大召回，小数据量可关闭
- **系统提示词**：可留空，也可自定义 RAG 提示词
- **对话窗口**：输入问题，按回车即可得到基于知识库的回答
- **检索来源**：回答下方会显示本次检索到的片段来源

### 3.3 ⚙️ 模型设置

- **当前模型提供者**：默认使用本地 GLM-4.5-Air
- **切换外部大模型**：选择 `external`，填入外部 OpenAI 兼容 API 地址、API Key 和模型名，点击“保存并切换”
  - 例如 **Kimi Code**：`https://api.kimi.com/coding/v1`，模型 `kimi-for-coding`
  - 例如 Moonshot：`https://api.moonshot.cn/v1`，模型 `kimi-latest`
  - 切换后，知识库问答和 RAG API 都会调用外部大模型
- 也展示当前后端配置：Embedding 模型目录、向量库和文件存储路径等

## 4. 服务管理（在 Linux 上执行）

知识库已配置为 systemd 用户服务，开机自动启动。

```bash
# 查看 Gradio Web UI 状态
systemctl --user status knowledge-base.service

# 查看 OpenAI 兼容 RAG API 状态
systemctl --user status knowledge-base-api.service

# 手动重启
systemctl --user restart knowledge-base.service
systemctl --user restart knowledge-base-api.service

# 停止
systemctl --user stop knowledge-base.service
systemctl --user stop knowledge-base-api.service

# 取消开机自启
systemctl --user disable knowledge-base.service
systemctl --user disable knowledge-base-api.service
```

项目目录：

```bash
~/ai-training/glm-4.5-air/knowledge-base
```

一键手动启动脚本：

```bash
cd ~/ai-training/glm-4.5-air/knowledge-base
./start.sh
```

## 5. 通过 API 使用（Cherry Studio / 其他客户端）

知识库同时暴露了 OpenAI 兼容的 RAG API：

```text
Base URL: http://192.168.10.2:7861/v1
API Key:  任意填写，例如 sk-local
模型名：   知识库名称，例如 系统文档
```

### Cherry Studio 配置示例

1. 打开 Cherry Studio → 设置 → 模型服务 → 添加自定义服务商
2. 填写：
   - **名称**：本地 RAG
   - **API 地址**：`http://192.168.10.2:7861/v1`
   - **API 密钥**：`sk-local`
   - **模型**：`系统文档`（或任意已有知识库名称）
3. 新建对话时选择该模型即可

> 如果该模型名称不是已有知识库，会自动 fallback 到第一个可用知识库。

### 通过 curl 调用

```bash
curl -X POST http://192.168.10.2:7861/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "系统文档",
    "messages": [{"role": "user", "content": "本地大模型地址是多少？"}],
    "stream": false
  }'
```

## 6. 常见问题

### 5.1 页面打不开？

1. 检查 Linux 工作站是否开机
2. 在 Linux 上执行 `systemctl --user status knowledge-base.service` 查看服务是否运行
3. 检查 Windows 能否 ping 通 `192.168.10.2`
4. 确认防火墙已放行 7860 端口

### 5.2 回答质量不好？

- 调整 **分块大小**：太小会丢失上下文，太大会降低检索精度
- 增加 **检索片段数量**：让更多上下文进入模型
- 开启 **查询扩展**：对复杂问题扩大召回
- 检查上传文件是否被正确解析（可在“文件列表”中查看）
- 上传更多相关文档以丰富知识库

### 5.3 上传失败或解析失败？

- 检查文件格式是否在支持列表内
- 对于加密或扫描版 PDF，OCR 结果可能不完美
- 查看服务日志：

```bash
journalctl --user -u knowledge-base.service -n 100
```

## 7. 数据存储位置

所有数据都保存在 Linux 工作站的本地磁盘上：

- 上传的原始文件：`~/ai-training/glm-4.5-air/knowledge-base/data/uploads/`
- 向量数据库：`~/ai-training/glm-4.5-air/knowledge-base/data/chroma/`
- BM25 关键词索引：`~/ai-training/glm-4.5-air/knowledge-base/data/indexes/`
- Embedding 模型：`~/ai-training/glm-4.5-air/knowledge-base/data/models/bge-m3-onnx/`
- 重排序模型：`~/ai-training/glm-4.5-air/knowledge-base/data/models/bge-reranker-v2-m3-onnx/`

## 8. 备注

- 本系统只负责**知识库后端**；复杂的代码执行、Agent 任务仍按你的需求放在 Windows 主机上处理。
- GLM-4.5-Air 大模型服务需单独保持运行。
- 重排序器、查询扩展、图片 OCR 都会增加单次请求的耗时，可根据数据量和硬件情况灵活开关。
