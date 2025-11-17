import gradio as gr
import time  # 仅用于模拟延迟，实际应替换为您的真实查询逻辑

# ==========================
# 模拟分阶段后端处理函数（使用 yield 流式输出）
# ==========================
def process_query_streaming(user_input):
    """
    模拟分阶段处理，并在每个阶段结束后 yield 当前状态和结果。
    返回格式: (top_k, graphrag, flow_state)
    flow_state: 0=初始, 1=中医完成, 2=西医完成, 3=整合完成, 4=全部完成
    """

    # 初始状态
    yield "", "", 0

    # 模拟中医查询
    time.sleep(0.8)  # 模拟耗时
    top_k_result = f"中医分析中... 基于症状 '{user_input}'，初步建议：\n- 疏肝理气\n- 养心安神"
    yield top_k_result, "", 1

    # 模拟西医查询
    time.sleep(0.8)
    graphrag_result = f"西医知识图谱匹配：\n- ICD-10: F51.0 (失眠)\n- 相关检查: 睡眠监测, 甲状腺功能"
    yield top_k_result, graphrag_result, 2

    # 模拟信息整合
    time.sleep(0.6)
    integrated_topk = f"【综合建议】\n{top_k_result}\n\n补充：{graphrag_result}"
    yield integrated_topk, graphrag_result, 3

    # 最终输出
    time.sleep(0.4)
    final_topk = f"✅ 最终诊断建议：\n{integrated_topk}"
    final_graphrag = f"✅ 知识图谱确认：\n{graphrag_result}"
    yield final_topk, final_graphrag, 4


# ==========================
# 生成流程图 HTML（根据状态高亮）
# ==========================
def render_flow_chart(state):
    colors = {
        0: "#cccccc",  # 灰色 - 未开始
        1: "#4CAF50",  # 绿色 - 中医完成
        2: "#2196F3",  # 蓝色 - 西医完成
        3: "#FF9800",  # 橙色 - 整合完成
        4: "#9C27B0",  # 紫色 - 全部完成
    }
    bg_colors = {
        0: "#f5f5f5",
        1: "#e6f7ff",
        2: "#e6f7ff",
        3: "#fff3e0",
        4: "#f3e5f5",
    }

    # 根据当前状态决定各节点颜色
    tcm_color = colors[1] if state >= 1 else colors[0]
    wm_color = colors[2] if state >= 2 else colors[0]
    merge_color = colors[3] if state >= 3 else colors[0]
    output_color = colors[4] if state >= 4 else colors[0]

    tcm_bg = bg_colors[1] if state >= 1 else bg_colors[0]
    wm_bg = bg_colors[2] if state >= 2 else bg_colors[0]
    merge_bg = bg_colors[3] if state >= 3 else bg_colors[0]
    output_bg = bg_colors[4] if state >= 4 else bg_colors[0]

    html = f"""
    <div style="display: flex; justify-content: space-around; align-items: center; margin: 15px 0;">
        <div style="text-align: center;">
            <div style="width: 80px; height: 80px; line-height: 80px; border: 2px solid {tcm_color}; border-radius: 50%; display: inline-block; background-color: {tcm_bg}; font-size: 14px; font-weight: {'bold' if state >= 1 else 'normal'};">中医查询</div>
        </div>
        <div style="font-size: 24px;">→</div>
        <div style="text-align: center;">
            <div style="width: 80px; height: 80px; line-height: 80px; border: 2px solid {wm_color}; border-radius: 50%; display: inline-block; background-color: {wm_bg}; font-size: 14px; font-weight: {'bold' if state >= 2 else 'normal'};">西医查询</div>
        </div>
        <div style="font-size: 24px;">→</div>
        <div style="text-align: center;">
            <div style="width: 80px; height: 80px; line-height: 80px; border: 2px solid {merge_color}; border-radius: 50%; display: inline-block; background-color: {merge_bg}; font-size: 14px; font-weight: {'bold' if state >= 3 else 'normal'};">整合信息</div>
        </div>
        <div style="font-size: 24px;">→</div>
        <div style="text-align: center;">
            <div style="width: 80px; height: 80px; line-height: 80px; border: 2px solid {output_color}; border-radius: 50%; display: inline-block; background-color: {output_bg}; font-size: 14px; font-weight: {'bold' if state >= 4 else 'normal'};">输出结果</div>
        </div>
    </div>
    """
    return html


# ==========================
# 主处理函数（generator，支持流式更新）
# ==========================
def respond_streaming(message, chat_history):
    if not message.strip():
        yield "", chat_history, "", "", gr.HTML(value=render_flow_chart(0))
        return

    # 添加用户消息
    new_history = chat_history + [{"role": "user", "content": message}]
    # 添加一个占位的 assistant 消息（后续会被更新）
    new_history = new_history + [{"role": "assistant", "content": "正在分析..."}]

    final_topk = ""
    final_graphrag = ""

    for topk, graphrag, state in process_query_streaming(message):
        final_topk = topk if topk else final_topk
        final_graphrag = graphrag if graphrag else final_graphrag

        # 实时更新所有组件
        current_chat = new_history.copy()
        if state > 0:
            current_chat[-1]["content"] = final_topk  # 更新对话框内容

        flow_html = render_flow_chart(state)

        yield (
            "",  # 清空输入框（仅在最后清空，这里保留也可）
            current_chat,
            final_topk,
            final_graphrag,
            gr.HTML(value=flow_html)
        )

    # 最终清空输入框
    yield "", current_chat, final_topk, final_graphrag, gr.HTML(value=render_flow_chart(4))


# ==========================
# 构建界面
# ==========================
with gr.Blocks(title="中医问诊辅助系统 - 流程可视化") as demo:
    gr.Markdown("## 🩺 中医问诊辅助系统（带实时流程图）")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=350):
            chatbot = gr.Chatbot(type="messages", height=550, label="问诊对话")
            msg = gr.Textbox(label="请输入您的症状或问题", placeholder="例如：失眠、乏力、食欲不振...")

        with gr.Column(scale=1, min_width=400):
            topk_output = gr.Textbox(label="💡 Top-K 推荐结果", interactive=False, lines=6, max_lines=10)
            graphrag_output = gr.Textbox(label="🧬 GraphRAG 知识图谱查询", interactive=False, lines=6, max_lines=15)
            flow_chart_display = gr.HTML(value=render_flow_chart(0))  # 初始状态

    # 使用 queue=True 启用流式输出
    msg.submit(
        respond_streaming,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, topk_output, graphrag_output, flow_chart_display],
        queue=True  # 关键：启用队列以支持 yield
    )

# 启动
if __name__ == "__main__":
    demo.queue()  # 启用队列
    demo.launch(inbrowser=True)