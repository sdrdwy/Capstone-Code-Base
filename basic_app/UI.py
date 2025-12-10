#!/usr/bin/env python3
"""
中西医结合问诊系统 - Gradio UI 版本
"""

import os
import sys
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_neo4j import Neo4jGraph
import gradio as gr

# 保持与 main.py 完全一致的相对导入
from .agents.west_agent import WestAgent
from .agents.tcm_agent import TcmAgent
from .agents.supervisor_agent import SupervisorAgent
from .agents.final_agent import FinalAgent
from .utils.query_fix import fix_query


# ==================== 全局状态封装 ====================
class AppState:
    def __init__(self):
        self.llm = None
        self.west_agent = None
        self.tcm_agent = None
        self.final_agent = None
        self.supervisor_agent = None
        self.tcm_vectorstore = None
        self.enable_advice = True
        self.conversation_ended = False
        self.last_tcm_response = {}
        self.last_west_response = {}
        self.last_supervisor_output = ""

    def initialize(self):
        load_dotenv()
        self.llm = ChatTongyi(model="qwen-max", temperature=0.1)
        graph = Neo4jGraph(database=os.environ["DB_NAME"])
        embedding = DashScopeEmbeddings(model="text-embedding-v2")
        embedding_v3 = DashScopeEmbeddings(model="text-embedding-v3")

        west_vectorstore = Chroma(
            persist_directory="./chroma_db_dash_w",
            embedding_function=embedding
        )
        tcm_vectorstore = Chroma(
            persist_directory="basic_app/chroma_db_embedding",
            embedding_function=embedding,
        )
        tcm_med_vectorstore = Chroma(
            persist_directory="chroma_TCM_rag_db_qwen",
            embedding_function=embedding_v3,
            collection_name="medical_book_qwen"
        )

        self.west_agent = WestAgent(llm=self.llm, retriever=west_vectorstore.as_retriever())
        self.tcm_agent = TcmAgent(llm=self.llm, graph=graph, retriever=tcm_med_vectorstore.as_retriever())
        self.final_agent = FinalAgent(llm=self.llm)
        self.supervisor_agent = SupervisorAgent(llm=self.llm)
        self.tcm_vectorstore = tcm_vectorstore

        self.conversation_ended = False
        self.last_tcm_response = {}
        self.last_west_response = {}
        self.last_supervisor_output = ""

    def reset(self):
        if self.final_agent:
            self.final_agent.reset_conversation()
        self.conversation_ended = False
        self.last_supervisor_output = ""
        self.last_tcm_response = {}
        self.last_west_response = {}


# 实例化全局状态
app_state = AppState()


# ==================== 核心交互函数 ====================
def format_docs(docs: List[Any]) -> str:
    if not docs:
        return "（无检索结果）"
    texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in docs]
    return "\n\n---\n\n".join(texts)


def process_user_input(user_input: str, chat_history: List[Tuple[str, str]]) -> Tuple[
    List[Tuple[str, str]], str, str, str, str, str, str, bool]:
    """
    处理用户输入并返回所有 UI 组件更新
    """
    if app_state.conversation_ended:
        gr.Info("当前问诊已结束，请点击“重置”开始新对话。")
        return chat_history, "", "", "", "", "", "", True

    if not user_input.strip():
        return chat_history, "", "", "", "", "", "", False

    try:
        # 1. Fix query
        fixed_query = fix_query(user_input, app_state.llm, app_state.tcm_vectorstore, 10)['query']

        # 2. 并行查询中西医 Agent
        west_response = app_state.west_agent.query(user_input,app_state.final_agent.conversation_history)
        tcm_response = app_state.tcm_agent.query(fixed_query,app_state.final_agent.conversation_history)

        app_state.last_west_response = west_response
        app_state.last_tcm_response = tcm_response

        # 3. FinalAgent 生成医生回复
        final_response = app_state.final_agent.process_input(
            patient_input=user_input,
            advice=app_state.last_supervisor_output if app_state.enable_advice else ""
        )
        doctor_reply = final_response['response']
        is_ended = final_response['is_ended']

        # 更新聊天历史
        chat_history.append((user_input, doctor_reply))

        # 4. Supervisor 评估
        conversation_history = "\n".join(app_state.final_agent.conversation_history)
        supervision_result = app_state.supervisor_agent.evaluate_conversation(
            conversation_history,
            tcm_response['result'],
            west_response['result']
        )

        if app_state.enable_advice and supervision_result.get('should_advise') and supervision_result.get('advice'):
            app_state.last_supervisor_output = supervision_result['advice']
        else:
            app_state.last_supervisor_output = ""

        # 5. 检查是否结束
        if is_ended:
            app_state.conversation_ended = True
            analysis = app_state.supervisor_agent.analyze_diagnosis_process(app_state.final_agent.conversation_history)
            chat_history.append((None, f"**【问诊结束】**\n\n{analysis}"))

        # 返回所有 UI 更新
        return (
            chat_history,
            app_state.last_supervisor_output or "(无建议)",
            west_response['result'],
            tcm_response['result'],
            # format_docs(tcm_response.get('retrieved_docs', [])),
            tcm_response.get('retrieved_docs', []),
            format_docs(west_response.get('retrieved_docs', [])),
            str(tcm_response.get('graph', '（无图谱结果）')),
            app_state.conversation_ended
        )

    except Exception as e:
        error_msg = f"❌ 错误: {str(e)}"
        chat_history.append((user_input, error_msg))
        return chat_history, error_msg, error_msg, error_msg, error_msg, error_msg, error_msg, True


def end_conversation(chat_history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], bool]:
    if app_state.conversation_ended:
        return chat_history, True
    if not app_state.final_agent.conversation_history:
        gr.Info("尚未开始问诊。")
        return chat_history, False

    analysis = app_state.supervisor_agent.analyze_diagnosis_process(app_state.final_agent.conversation_history)
    chat_history.append((None, f"**【强制结束 - 问诊总结】**\n\n{analysis}"))
    app_state.conversation_ended = True
    return chat_history, True


def toggle_advice(enable: bool) -> bool:
    app_state.enable_advice = enable
    return enable


def reset_conversation() -> Tuple[List, str, str, str, str, str, str, bool, bool]:
    app_state.reset()
    return [], "", "", "", "", "", "", False, False


# ==================== Gradio UI 构建 ====================
with gr.Blocks(title="中西医结合智能问诊系统") as demo:
    gr.Markdown("## 🏥 中西医结合智能问诊系统")
    gr.Markdown("输入您的症状，系统将并行调用中西医知识库，并由 AI 医生逐步问诊。")

    with gr.Row():
        # ========== 左侧：检索结果 ==========
        with gr.Column(scale=2):
            gr.Markdown("### 📚 中医知识库 (RAG)")
            tcm_rag_output = gr.Textbox(label="中医 RAG 检索结果", lines=10, interactive=False)
            gr.Markdown("### 📚 西医知识库 (RAG)")
            west_rag_output = gr.Textbox(label="西医 RAG 检索结果", lines=10, interactive=False)
            gr.Markdown("### 🧠 中医 GraphRAG")
            tcm_graph_output = gr.Textbox(label="中医图谱查询结果", lines=10, interactive=False)

        # ========== 中间：Chatbot ==========
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=600,
                avatar_images=("assets/patient.png", "assets/doctor.png"),  # 可选
                show_label=False
            )
            user_input = gr.Textbox(
                placeholder="请输入您的症状，例如：'最近总是头晕乏力'...",
                label="患者输入",
                container=False
            )
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary")
                reset_btn = gr.Button("重置对话")

        # ========== 右侧：Agent 输出 ==========
        with gr.Column(scale=2):
            gr.Markdown("### 👨‍🏫 专家建议 (Supervisor)")
            supervisor_output = gr.Textbox(label="建议内容", lines=4, interactive=False)
            gr.Markdown("### 🩺 西医 Agent 输出")
            west_output = gr.Textbox(label="西医分析", lines=8, interactive=False)
            gr.Markdown("### 🌿 中医 Agent 输出")
            tcm_output = gr.Textbox(label="中医分析", lines=8, interactive=False)

    # ========== 底部控制 ==========
    with gr.Row():
        end_btn = gr.Button("结束对话并总结", variant="stop")
        advice_toggle = gr.Checkbox(label="启用专家建议", value=True)

    # ========== 状态变量 ==========
    conversation_ended_state = gr.State(False)

    # ========== 事件绑定 ==========
    submit_event = user_input.submit(
        fn=process_user_input,
        inputs=[user_input, chatbot],
        outputs=[
            chatbot,
            supervisor_output,
            west_output,
            tcm_output,
            tcm_rag_output,
            west_rag_output,
            tcm_graph_output,
            conversation_ended_state
        ],
        show_progress="full"
    ).then(lambda: "", None, user_input)  # 清空输入框

    submit_btn.click(
        fn=process_user_input,
        inputs=[user_input, chatbot],
        outputs=[
            chatbot,
            supervisor_output,
            west_output,
            tcm_output,
            tcm_rag_output,
            west_rag_output,
            tcm_graph_output,
            conversation_ended_state
        ],
        show_progress="full"
    ).then(lambda: "", None, user_input)

    end_btn.click(
        fn=end_conversation,
        inputs=[chatbot],
        outputs=[chatbot, conversation_ended_state]
    )

    advice_toggle.change(
        fn=toggle_advice,
        inputs=advice_toggle,
        outputs=advice_toggle
    )

    reset_btn.click(
        fn=reset_conversation,
        inputs=[],
        outputs=[
            chatbot,
            supervisor_output,
            west_output,
            tcm_output,
            tcm_rag_output,
            west_rag_output,
            tcm_graph_output,
            conversation_ended_state,
            advice_toggle
        ]
    )

    # ========== 初始化 ==========
    demo.load(
        fn=lambda: None,
        inputs=None,
        outputs=None,
        # _js="() => { document.title = '中西医结合问诊系统'; }"
    )


# ==================== 启动入口 ====================
if __name__ == "__main__":
    # 与 main.py 一致的参数处理
    enable_advice_default = True
    if "--disable" in sys.argv:
        enable_advice_default = False

    app_state.enable_advice = enable_advice_default
    app_state.initialize()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )