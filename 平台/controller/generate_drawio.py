#!/usr/bin/env python3
"""生成 Controller V0.1 架构图的 draw.io 文件（未压缩 XML）"""
import xml.etree.ElementTree as ET
from xml.dom import minidom


def create_drawio(filename: str):
    mxfile = ET.Element("mxfile")
    mxfile.set("host", "app.diagrams.net")
    mxfile.set("modified", "2026-06-26T00:00:00.000Z")
    mxfile.set("agent", "controller-arch-generator")
    mxfile.set("etag", "v0.1")
    mxfile.set("version", "24.0.0")
    mxfile.set("type", "device")

    diagram = ET.SubElement(mxfile, "diagram")
    diagram.set("name", "Controller-V0.1")
    diagram.set("id", "controller-v0-1")

    model = ET.SubElement(diagram, "mxGraphModel")
    model.set("dx", "1200")
    model.set("dy", "800")
    model.set("grid", "1")
    model.set("gridSize", "10")
    model.set("guides", "1")
    model.set("tooltips", "1")
    model.set("connect", "1")
    model.set("arrows", "1")
    model.set("fold", "1")
    model.set("page", "1")
    model.set("pageScale", "1")
    model.set("pageWidth", "1600")
    model.set("pageHeight", "1200")
    model.set("math", "0")
    model.set("shadow", "0")

    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    cells = root
    _id = [100]

    def node(x, y, w, h, value, style, id=None):
        cid = id or str(_id[0])
        _id[0] += 1
        cell = ET.SubElement(cells, "mxCell")
        cell.set("id", cid)
        cell.set("value", value)
        cell.set("style", style)
        cell.set("vertex", "1")
        cell.set("parent", "1")
        geo = ET.SubElement(cell, "mxGeometry")
        geo.set("x", str(x))
        geo.set("y", str(y))
        geo.set("width", str(w))
        geo.set("height", str(h))
        geo.set("as", "geometry")
        return cid

    def edge(src, tgt, value="", style=None):
        cid = str(_id[0])
        _id[0] += 1
        if style is None:
            style = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#666666;"
        cell = ET.SubElement(cells, "mxCell")
        cell.set("id", cid)
        cell.set("value", value)
        cell.set("style", style)
        cell.set("edge", "1")
        cell.set("parent", "1")
        cell.set("source", src)
        cell.set("target", tgt)
        geo = ET.SubElement(cell, "mxGeometry")
        geo.set("relative", "1")
        geo.set("as", "geometry")
        return cid

    # Styles
    box = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontColor=#333333;"
    box_blue = "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;"
    box_green = "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;"
    box_orange = "rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;"
    box_red = "rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;"
    box_purple = "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;"
    box_yellow = "rounded=1;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;"
    cylinder = "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;fillColor=#e1d5e7;strokeColor=#9673a6;"
    blackbox = "rounded=1;whiteSpace=wrap;html=1;fillColor=#1a1a1a;strokeColor=#000000;fontColor=#ffffff;"
    diamond = "rhombus;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;"
    swimlane = "swimlane;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;startSize=30;"

    # Title
    node(40, 10, 300, 30, "Controller V0.1 认知操作系统架构", "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=18;fontStyle=1")

    # Input
    n_input = node(700, 70, 140, 50, "用户输入", box + "fontSize=14;fontStyle=1")

    # Query Expansion
    n_qe = node(700, 160, 160, 60, "查询扩写模块\nQueryExpander", box_purple + "fontSize=12;fontStyle=1")
    edge(n_input, n_qe)

    # Context memory (short-term)
    n_ctx = node(940, 155, 140, 70, "短期记忆\n上下文", box_yellow + "dashed=1;")
    edge(n_qe, n_ctx, value="参考")

    # KB Retrieval lane container
    kb_swim = node(40, 260, 700, 260, "KB 检索", swimlane + "fontStyle=1")
    # child nodes need parent=kb_swim, but for simplicity place inside and set parent? We'll use absolute coordinates and parent=1, visually inside.
    n_kb_normal = node(80, 320, 140, 60, "正常查询", box_blue)
    n_kb_syn = node(260, 320, 140, 60, "同义词/近义词\n查询", box_blue)
    n_kb_expand = node(440, 320, 160, 60, "扩写查询", box_blue)
    n_kb_ctx = node(640, 320, 140, 60, "上下文思考", box_yellow)
    n_kb_search = node(320, 420, 200, 60, "KB 向量检索\n(BGE-M3 + BM25 + rerank)", box + "fontSize=12;fontStyle=1")
    n_kb_res = node(340, 520, 160, 60, "KB 检索结果\n(score)", box_green + "fontSize=12;fontStyle=1")

    edge(n_qe, n_kb_normal, style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#6c8ebf;")
    edge(n_qe, n_kb_syn, style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#6c8ebf;")
    edge(n_qe, n_kb_expand, style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#6c8ebf;")
    edge(n_kb_normal, n_kb_search)
    edge(n_kb_syn, n_kb_search)
    edge(n_kb_expand, n_kb_search)
    edge(n_kb_ctx, n_kb_search, value="结合")
    edge(n_ctx, n_kb_ctx, style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#d6b656;dashed=1;")
    edge(n_kb_search, n_kb_res)

    # LLM Retrieval lane container
    llm_swim = node(820, 260, 740, 260, "LLM 检索", swimlane + "fontStyle=1")
    n_llm_hyb = node(860, 320, 160, 60, "Hybrid + Expanded\n查询生成", box_blue)
    n_llm_ctx = node(1060, 320, 140, 60, "上下文思考", box_yellow)
    n_llm_svm = node(1240, 320, 180, 60, "SVM-Linear\n分类器", box_blue + "fontSize=12;fontStyle=1")
    n_llm_res = node(1080, 520, 180, 60, "LLM 检测结果\n(known probability)", box_green + "fontSize=12;fontStyle=1")

    edge(n_qe, n_llm_hyb, style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#9673a6;")
    edge(n_llm_hyb, n_llm_ctx)
    edge(n_llm_ctx, n_llm_svm, value="结合")
    edge(n_ctx, n_llm_ctx, style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#d6b656;dashed=1;")
    edge(n_llm_svm, n_llm_res)

    # Classifier
    n_class = node(740, 560, 180, 70, "四象限分类器\nKB score × LLM prob", diamond + "fontSize=12;fontStyle=1")
    edge(n_kb_res, n_class, value="KB score")
    edge(n_llm_res, n_class, value="LLM prob")

    # Case A
    n_a = node(180, 700, 180, 70, "A. KB高 + LLM高\nRAG 检索直接生成", box_green + "fontSize=12;fontStyle=1")
    edge(n_class, n_a, value="A")
    n_a_mem = node(180, 820, 200, 50, "不对记忆做任何改动", box + "fontStyle=1")
    edge(n_a, n_a_mem)

    # Case B
    n_b = node(460, 700, 220, 70, "B. KB高 + LLM低\n依 KB 做逻辑推理", box_blue + "fontSize=12;fontStyle=1")
    edge(n_class, n_b, value="B")
    n_b_shadow = node(460, 820, 220, 60, "多路影子验证\n（记录学习样本）", box_yellow)
    n_b_lora = node(460, 930, 220, 60, "后训练调参\nfrozen LoRA", box_purple)
    edge(n_b, n_b_shadow)
    edge(n_b_shadow, n_b_lora)

    # Case C
    n_c = node(760, 700, 240, 70, "C. KB低 + LLM高\n启动多路径 MoE 推理", box_orange + "fontSize=12;fontStyle=1")
    edge(n_class, n_c, value="C")
    n_c_moe = node(760, 820, 240, 60, "多路径推理 + 网查对比", box_orange)
    n_c_verify = node(760, 920, 240, 60, "验证结论合理性", box_yellow)
    n_c_write = node(760, 1020, 240, 60, "写入 KB\n（长期显式记忆）", box_green)
    edge(n_c, n_c_moe)
    edge(n_c_moe, n_c_verify)
    edge(n_c_verify, n_c_write, value="通过")

    # Case D
    n_d = node(1100, 700, 240, 70, "D. KB低 + LLM低\n联网补充 KB", box_red + "fontSize=12;fontStyle=1")
    edge(n_class, n_d, value="D")
    n_d_web = node(1100, 820, 240, 60, "联网搜索摘要\n补充进 KB 库", box_red)
    n_d_to_b = node(1100, 920, 240, 60, "进入 B 流程\n整理/学习/调参", box_blue)
    edge(n_d, n_d_web)
    edge(n_d_web, n_d_to_b)

    # Memory legend
    node(1380, 700, 160, 70, "短期记忆\n（上下文）", box_yellow + "dashed=1;fontSize=11;")
    node(1380, 800, 160, 70, "长期显式记忆\n（KB）", cylinder + "fontSize=11;")
    node(1380, 900, 160, 80, "长期隐式记忆\n（LLM 内部权重黑箱）", blackbox + "fontSize=11;")

    # Output
    n_out = node(740, 1140, 180, 60, "最终回答 / 输出", box + "fontSize=14;fontStyle=1")
    edge(n_a_mem, n_out, style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#666666;")
    edge(n_b_lora, n_out)
    edge(n_c_write, n_out)
    edge(n_d_to_b, n_out)

    # Write file
    rough = ET.tostring(mxfile, encoding="utf-8")
    reparsed = minidom.parseString(rough)
    pretty = reparsed.toprettyxml(indent="  ", encoding="utf-8")
    with open(filename, "wb") as f:
        f.write(pretty)
    print(f"已生成: {filename}")


if __name__ == "__main__":
    create_drawio("architecture.drawio")
