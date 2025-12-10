#!/usr/bin/env python3
"""
中西医结合问诊系统 Gradio Web UI
"""

import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_neo4j import Neo4jGraph

from .agents.west_agent import WestAgent, medical_qa_pipeline
from .agents.tcm_agent import TcmAgent
from .agents.supervisor_agent import SupervisorAgent
from .agents.final_agent import FinalAgent
from .utils.query_fix import fix_query


# ======================
# 初始化一次（避免每次推理都初始化）
# ======================
load_dotenv()

llm = ChatTongyi(model="qwen-max", temperature=0.1)

embedding = DashScopeEmbeddings(model="text-embedding-v2")

west_vectorstore = Chroma(
    persist_directory="./chroma_db_dash_w",
    embedding_function=embedding
)

tcm_vectorstore = Chroma(
    persist_directory="basic_app/chroma_db_embedding",
    embedding_function=embedding
)

graph = Neo4jGraph(database=os.environ["DB_NAME"])

west_agent = WestAgent(
    llm=llm,
    retriever=west_vectorstore.as_retriever(search_kwargs={"k": 3})
)

tcm_agent = TcmAgent(llm=llm, graph=graph)

final_agent = FinalAgent(llm=llm, west_agent=west_agent, tcm_agent=tcm_agent)

supervisor_agent = SupervisorAgent(llm=llm)


# ======================
# Gradio 交互函数
# ======================
def reset_conversation():
    final_agent.reset_conversation()
    return [], "", "", "", ""


def send_message(history, user_input, _):
    if not user_input.strip():
        return history, "", "", "", ""

    # 保存原始输入
    patient_input = user_input.strip()

    # === 西医 Agent ===
    west_response = "无结果"
    try:
        west_result = medical_qa_pipeline(
            llm_choice="qwen-max",
            vector_db_path="./chroma_db_dash_w",
            user_query=patient_input
        )
        west_response = west_result.get('answer', '无结果')
    except Exception as e:
        west_response = f"⚠️ 西医错误: {str(e)}"

    # === 中医 Agent ===
    tcm_response = "无结果"
    try:
        fixed_query_result = fix_query(patient_input, llm, tcm_vectorstore, 10)
        fixed_query = fixed_query_result['query']
        if len(fixed_query) > 100:
            fixed_query = fixed_query[:100]
        tcm_result = tcm_agent.query(fixed_query)
        tcm_response = tcm_result.get('result', '无结果')
    except Exception as e:
        tcm_response = f"⚠️ 中医错误: {str(e)}"

    # === Final Agent 处理 ===
    final_response = final_agent.process_input(
        patient_input=patient_input,
        west_response=west_response,
        tcm_response=tcm_response
    )
    doctor_reply = final_response['response']
    is_ended = final_response['is_ended']

    # 更新聊天历史
    history.append([patient_input, doctor_reply])

    # === Supervisor 评估 ===
    conversation_history = "\n".join(final_agent.conversation_history)
    supervision = supervisor_agent.evaluate_conversation(conversation_history)
    supervisor_output = supervision.get('advice', '') if supervision.get('should_advise') else ""

    if is_ended:
        # 触发总结（显示在 supervisor 区域或单独弹出）
        summary = final_agent.analyze_diagnosis_process()
        supervisor_output = f"【问诊总结】\n\n{summary}"

    return history, supervisor_output, west_response, tcm_response, ""


def end_diagnosis(history):
    if not final_agent.conversation_history:
        return history, "无问诊记录可总结。", "", "", ""

    summary = final_agent.analyze_diagnosis_process()
    supervisor_output = f"【问诊总结】\n\n{summary}"
    # 可选择不清空，或重置
    # final_agent.reset_conversation()
    return history, supervisor_output, "", "", ""


# ======================
# 构建 Gradio 界面
# ======================
with gr.Blocks(title="中西医结合智能问诊系统") as demo:
    gr.Markdown("## 🩺 中西医结合智能问诊系统")
    gr.Markdown("请输入您的症状或问题，系统将并行调用中西医知识库进行分析。")

    with gr.Row():
        # 左侧：聊天窗口
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="问诊对话",
                height=500,
                bubble_full_width=False
            )
            user_input = gr.Textbox(
                label="您的症状或问题",
                placeholder="例如：我最近头痛、乏力...",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("发送")
                reset_btn = gr.Button("重置对话")
                end_btn = gr.Button("结束问诊", variant="stop")

        # 右侧：三栏信息
        with gr.Column(scale=1):
            supervisor_box = gr.Textbox(
                label="🧑‍🏫 Supervisor 建议 / 问诊总结",
                interactive=False,
                lines=6
            )
            west_box = gr.Textbox(
                label="西医 Agent 输出",
                interactive=False,
                lines=6
            )
            tcm_box = gr.Textbox(
                label="中医 Agent 输出",
                interactive=False,
                lines=6
            )

    # 状态管理：不需要额外 state，final_agent 本身持有状态
    # 但为了兼容性，可留空 gr.State()

    # 事件绑定
    submit_event = submit_btn.click(
        fn=send_message,
        inputs=[chatbot, user_input],
        outputs=[chatbot, supervisor_box, west_box, tcm_box, user_input],
        queue=False
    )
    user_input.submit(
        fn=send_message,
        inputs=[chatbot, user_input],
        outputs=[chatbot, supervisor_box, west_box, tcm_box, user_input],
        queue=False
    )

    reset_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[chatbot, supervisor_box, west_box, tcm_box, user_input],
        queue=False
    )

    end_btn.click(
        fn=end_diagnosis,
        inputs=[chatbot],
        outputs=[chatbot, supervisor_box, west_box, tcm_box, user_input],
        queue=False
    )

if __name__ == "__main__":
    demo.launch()