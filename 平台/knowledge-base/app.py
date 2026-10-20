"""Gradio web UI for the enhanced local knowledge base."""
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr

import config
from knowledge_base import KnowledgeBaseManager
from llm import build_rag_messages, chat, format_sources, switch_provider, current_provider_info
from llm_adapter import adapter
from query_understanding import is_chitchat

kb_manager = KnowledgeBaseManager()


def refresh_kb_choices() -> Tuple[gr.update, gr.update]:
    choices = kb_manager.list_kbs()
    choices_with_empty = [""] + choices
    return (
        gr.update(choices=choices_with_empty, value=""),
        gr.update(choices=choices_with_empty, value=""),
    )


def load_provider_info() -> str:
    info = current_provider_info()
    if not info:
        return "当前没有可用的模型提供者"
    return (
        f"当前提供者：{info['provider']}\n"
        f"Base URL：{info['base_url']}\n"
        f"模型：{info['model']}"
    )


def apply_provider_settings(
    provider: str,
    base_url: str,
    api_key: str,
    model: str,
) -> Tuple[str, str]:
    if provider == "external":
        if not base_url or not model:
            return "外部 API 地址和模型名称不能为空", load_provider_info()
        adapter.add_provider("external", base_url.strip(), api_key.strip(), model.strip())
        switch_provider("external")
        return f"已切换到外部模型：{model}", load_provider_info()
    else:
        switch_provider("local")
        return "已切换到本地 GLM-4.5-Air", load_provider_info()


def handle_upload(
    kb_name: str,
    files: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[str, gr.update, gr.update]:
    if not kb_name or not kb_name.strip():
        return ("请输入知识库名称",) + refresh_kb_choices()
    if not files:
        return ("请先上传文件",) + refresh_kb_choices()

    file_paths = [Path(f) for f in files]
    status = kb_manager.add_files(
        kb_name=kb_name.strip(),
        file_paths=file_paths,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return (status,) + refresh_kb_choices()


def list_kb_files(kb_name: str) -> str:
    if not kb_name:
        return ""
    files = kb_manager.list_files(kb_name)
    if not files:
        return "该知识库暂无文件"
    return "\n".join(f"- {f}" for f in files)


def delete_kb(kb_name: str) -> Tuple[str, gr.update, gr.update]:
    if not kb_name:
        return "请先选择知识库", *refresh_kb_choices()
    status = kb_manager.delete_kb(kb_name)
    return status, *refresh_kb_choices()


def respond(
    message: str,
    history: List[Dict],
    kb_name: str,
    top_k: int,
    system_prompt: str,
    use_hybrid: bool,
    use_rerank: bool,
    use_query_expansion: bool,
):
    user_msg = {"role": "user", "content": message}
    base_history = list(history) + [user_msg]

    if not kb_name:
        yield base_history + [{"role": "assistant", "content": "请先选择一个知识库"}], ""
        return

    retrieved_chunks = []

    try:
        if is_chitchat(message):
            messages = [
                {"role": "system", "content": "你是一个友好的 AI 助手。"},
                {"role": "user", "content": message},
            ]
            sources = ""
        else:
            retrieved_chunks = kb_manager.search(
                kb_name=kb_name,
                query=message,
                top_k=top_k,
                use_hybrid=use_hybrid,
                use_rerank=use_rerank,
                use_query_expansion=use_query_expansion,
            )
            if not retrieved_chunks:
                yield base_history + [{"role": "assistant", "content": "未在知识库中找到相关内容。"}], ""
                return
            messages = build_rag_messages(message, retrieved_chunks, system_prompt)
            sources = format_sources(retrieved_chunks)

        stream = chat(messages, stream=True)
    except Exception as e:
        yield base_history + [{"role": "assistant", "content": f"调用失败: {e}"}], ""
        return

    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        partial += delta
        yield base_history + [{"role": "assistant", "content": partial}], ""

    yield base_history + [{"role": "assistant", "content": partial}], sources


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="本地 AI 知识库") as demo:
        gr.Markdown("# 🧠 本地 AI 知识库\n基于 Gradio + Chroma + BGE-M3 + GLM-4.5-Air（增强版）")

        kb_choices = [""] + kb_manager.list_kbs()

        with gr.Tabs():
            # ---------- Management tab ----------
            with gr.Tab("📚 知识库管理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        kb_name_input = gr.Textbox(
                            label="知识库名称",
                            placeholder="例如：技术文档、论文集、项目资料",
                        )
                        chunk_size = gr.Slider(
                            100, 2000, value=500, step=50, label="分块大小（字符）"
                        )
                        chunk_overlap = gr.Slider(
                            0, 500, value=50, step=10, label="分块重叠（字符）"
                        )
                        file_input = gr.File(
                            label="拖拽文件到此处上传（支持 txt/md/pdf/docx/csv/xlsx/图片）",
                            file_count="multiple",
                            file_types=[
                                ".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx",
                                ".json", ".py", ".js", ".ts", ".html", ".xml",
                                ".jpg", ".jpeg", ".png", ".webp", ".bmp",
                            ],
                        )
                        upload_btn = gr.Button("上传并建立索引", variant="primary")
                        status_text = gr.Textbox(label="处理状态", interactive=False, lines=6)

                    with gr.Column(scale=1):
                        gr.Markdown("### 已有知识库")
                        kb_select_manage = gr.Dropdown(
                            label="选择知识库", choices=kb_choices, value="", allow_custom_value=True
                        )
                        refresh_btn = gr.Button("刷新列表")
                        file_list = gr.Textbox(label="文件列表", interactive=False, lines=10)
                        del_kb_btn = gr.Button("删除知识库", variant="stop")

            # ---------- Chat tab ----------
            with gr.Tab("💬 知识库问答"):
                with gr.Row():
                    kb_select_chat = gr.Dropdown(
                        label="选择要使用的知识库",
                        choices=kb_choices,
                        value="",
                        allow_custom_value=True,
                        scale=4,
                    )
                    refresh_chat_kb_btn = gr.Button("🔄 刷新", scale=1)
                with gr.Row():
                    top_k = gr.Slider(1, 20, value=5, step=1, label="检索片段数量")
                    use_hybrid = gr.Checkbox(value=True, label="混合检索（向量+关键词）")
                    use_rerank = gr.Checkbox(value=True, label="重排序")
                    use_query_expansion = gr.Checkbox(value=False, label="查询扩展")
                system_prompt = gr.Textbox(
                    label="系统提示词（可自定义）",
                    value="",
                    placeholder="留空使用默认提示词",
                    lines=2,
                )

                chatbot = gr.Chatbot(label="对话", height=500)
                msg = gr.Textbox(label="输入问题", placeholder="请输入问题，按回车发送")
                with gr.Row():
                    clear = gr.Button("清空对话")
                sources_text = gr.Textbox(
                    label="检索来源", interactive=False, lines=5
                )

        # Auto-populate dropdowns when the page loads
        demo.load(fn=refresh_kb_choices, outputs=[kb_select_manage, kb_select_chat])

        # Bind events
        refresh_chat_kb_btn.click(
            fn=refresh_kb_choices,
            outputs=[kb_select_manage, kb_select_chat],
        )
        upload_btn.click(
            fn=handle_upload,
            inputs=[kb_name_input, file_input, chunk_size, chunk_overlap],
            outputs=[status_text, kb_select_manage, kb_select_chat],
        )
        refresh_btn.click(
            fn=refresh_kb_choices,
            outputs=[kb_select_manage, kb_select_chat],
        )
        kb_select_manage.change(
            fn=list_kb_files,
            inputs=kb_select_manage,
            outputs=file_list,
        )
        del_kb_btn.click(
            fn=delete_kb,
            inputs=kb_select_manage,
            outputs=[status_text, kb_select_manage, kb_select_chat],
        )
        msg.submit(
            fn=respond,
            inputs=[
                msg,
                chatbot,
                kb_select_chat,
                top_k,
                system_prompt,
                use_hybrid,
                use_rerank,
                use_query_expansion,
            ],
            outputs=[chatbot, sources_text],
        )
        clear.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, sources_text],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.queue()
    demo.launch(
        server_name=config.HOST,
        server_port=config.PORT,
        share=False,
        show_error=True,
    )
