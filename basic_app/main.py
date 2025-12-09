#!/usr/bin/env python3
"""
中西医结合问诊系统命令行应用

功能流程：
1. 等待用户输入症状或问题
2. 并行进行west_agent和tcm_agent的查询
3. 输入汇入final_agent和supervisor_agent
4. final_agent作为医生一步步问诊病人
5. supervisor_agent检测对话，决定是否给出建议
6. 在对话结束后，给出总结、打分和改进建议
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_neo4j import Neo4jGraph

from .agents.west_agent import WestAgent, medical_qa_pipeline
from .agents.tcm_agent import TcmAgent
from .agents.tcm_rag_agent import TcmRagAgent
from .agents.supervisor_agent import SupervisorAgent
from .agents.final_agent import FinalAgent
from .utils.query_fix import fix_query


def initialize_components():
    """初始化所有组件"""
    print("正在初始化系统组件...")
    
    # 加载环境变量
    load_dotenv()
    
    # 初始化LLM
    llm = ChatTongyi(
        model="qwen-max",        
        temperature=0.1,
    )
    
    # 初始化图数据库
    graph = Neo4jGraph(database=os.environ["DB_NAME"])
    
    # 初始化向量数据库
    embedding = DashScopeEmbeddings(model="text-embedding-v2")
    west_vectorstore = Chroma(
        persist_directory="./chroma_db_dash_w",
        embedding_function=embedding
    )
    tcm_vectorstore = Chroma(
        persist_directory="basic_app/chroma_db_embedding",
        embedding_function=embedding
    )
    
    tcm_rag_vectorstore = Chroma(
        persist_directory="./chroma_TCM_rag_db_qwen",
        embedding_function=embedding
    )

    # 初始化各Agent
    west_agent = WestAgent(
        llm=llm,
        retriever=west_vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    tcm_agent = TcmAgent(
        llm=llm,
        graph=graph
    )
    
    # 初始化中医RAG Agent
    tcm_rag_agent = TcmRagAgent(
        llm=llm,
        retriever=tcm_rag_vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    # final_agent不再需要west_agent和tcm_agent
    final_agent = FinalAgent(
        llm=llm
    )
    
    # Pass the agents to supervisor_agent so it can call them
    supervisor_agent = SupervisorAgent(llm=llm)
    # Store references to the agents so supervisor can call them
    supervisor_agent.west_agent = west_agent
    supervisor_agent.tcm_agent = tcm_agent
    supervisor_agent.tcm_rag_agent = tcm_rag_agent
    
    print("系统初始化完成！\n")
    
    return llm, west_agent, tcm_agent, final_agent, supervisor_agent, tcm_vectorstore, tcm_rag_agent


def run_diagnosis_system():
    """运行诊断系统主循环"""
    print("="*60)
    print("欢迎使用中西医结合智能问诊系统")
    print("="*60)
    print("提示：输入 'quit' 或 'exit' 退出系统")
    print("输入 'reset' 重置对话")
    print("-"*60)
    
    # 询问用户是否希望看到专家建议
    show_supervisor_advice = input("是否希望看到专家的建议？(y/n，默认为y): ").strip().lower()
    show_supervisor_advice = show_supervisor_advice in ['y', 'yes', '是', 'Y', '']
    
    # 初始化组件
    llm, west_agent, tcm_agent, final_agent, supervisor_agent, tcm_vectorstore, tcm_rag_agent = initialize_components()
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n患者: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n感谢使用中西医结合问诊系统，祝您健康！")
                break
            elif user_input.lower() == 'reset':
                final_agent.reset_conversation()
                print("对话已重置。")
                continue
            elif not user_input:
                continue
            
            print("\n正在分析您的症状...")
            
            # 并行执行西医和中医查询，但要处理可能的异常
            print("🔍 正在进行西医知识检索...")
            west_response = "无结果"  # 默认值
            try:
                west_result = medical_qa_pipeline(
                    llm_choice="qwen-max",
                    vector_db_path="./chroma_db_dash_w",
                    user_query=user_input
                )
                west_response = west_result['answer']
            except Exception as e:
                print(f"⚠️ 西医agent出现错误: {str(e)}，使用默认结果")
                west_response = "无结果"
            
            print("🌿 正在进行中医知识图谱查询...")
            tcm_response = "无结果"  # 默认值
            try:
                # 首先修复查询
                fixed_query_result = fix_query(user_input, llm, tcm_vectorstore, 10)
                fixed_query = fixed_query_result['query']
                
                # 限制查询语句长度
                if len(fixed_query) > 100:  # 限制为100字符
                    print("⚠️ 查询语句过长，已截断")
                    fixed_query = fixed_query[:100]
                
                # 然后进行图谱查询
                tcm_result = tcm_agent.query(fixed_query)
                tcm_response = tcm_result['result']
            except Exception as e:
                print(f"⚠️ 中医agent出现错误: {str(e)}，使用默认结果")
                tcm_response = "无结果"
            
            print("🌿 正在进行中医RAG检索...")
            tcm_rag_response = "无结果"  # 默认值
            try:
                # 使用tcm_rag_agent进行检索
                tcm_rag_result = tcm_rag_agent.query(user_input)
                tcm_rag_response = tcm_rag_result['answer']
            except Exception as e:
                print(f"⚠️ 中医RAG agent出现错误: {str(e)}，使用默认结果")
                tcm_rag_response = "无结果"
            
            # 合并中医知识图谱和RAG的结果
            combined_tcm_response = f"知识图谱结果：{tcm_response}\nRAG结果：{tcm_rag_response}"
            
            print("✅ 分析完成，正在整合信息...")
            
            conversation_history = "\n".join(final_agent.conversation_history)
            
            # supervisor实时提供问诊辅助
            realtime_assistance = supervisor_agent.provide_realtime_assistance(
                patient_input=user_input,
                conversation_history=conversation_history
            )
            
            # 根据开关决定final_agent是否能理解建议（即是否传递给final_agent）
            if not show_supervisor_advice:
                # 如果开关关闭，则final_agent接收不到建议
                realtime_assistance = None
            
            # 交给final_agent处理
            final_response = final_agent.process_input(
                patient_input=user_input,
                supervisor_advice=realtime_assistance
            )
            
            # 获取医生回复
            doctor_response = final_response['response']
            print(f"\n👨‍⚕️ 医生: {doctor_response}")
            
            # supervisor默认总是调用tcm和west_agent来获取额外信息
            should_call_west = supervisor_agent.always_call_west_agent(conversation_history + f"\n患者最新输入: {user_input}")
            should_call_tcm = supervisor_agent.always_call_tcm_agent(conversation_history + f"\n患者最新输入: {user_input}")
            
            additional_info = []
            if should_call_west:
                print("🔍 专家正在调用西医知识库获取更多信息...")
                west_additional = supervisor_agent.call_west_agent(user_input)
                additional_info.append(f"西医建议: {west_additional}")
            
            if should_call_tcm:
                print("🌿 专家正在调用中医知识库获取更多信息...")
                tcm_additional = supervisor_agent.call_tcm_agent(user_input)
                additional_info.append(f"中医建议: {tcm_additional}")
                
            # 检查是否需要调用tcm_rag_agent
            should_call_tcm_rag = len(conversation_history) > 50  # 假设对话历史较长时需要额外的RAG信息
            if should_call_tcm_rag:
                print("🌿 专家正在调用中医RAG知识库获取更多信息...")
                tcm_rag_additional = supervisor_agent.call_tcm_rag_agent(user_input)
                additional_info.append(f"中医RAG建议: {tcm_rag_additional}")
            
            # 显示额外信息（如果需要）
            if additional_info and show_supervisor_advice:
                for info in additional_info:
                    print(f"\n🔬 {info}")
            
            # 检查是否结束对话
            if final_response['is_ended']:
                print("" + "="*60)
                print("问诊结束")
                print("="*60)
                
                # 生成诊断总结
                summary = supervisor_agent.generate_final_summary(conversation_history)
                print(f"📋 问诊总结报告：")
                print(summary)
                
                # 如果用户选择查看专家建议，也显示分析
                if show_supervisor_advice:
                    analysis = supervisor_agent.analyze_diagnosis_process(conversation_history)
                    print(f"🔍 专家分析与评价：")
                    print(analysis)
                
                # 询问是否开始新对话
                continue_diag = input("是否开始新的问诊？(y/n): ").strip().lower()
                if continue_diag not in ['y', 'yes', '是', 'Y']:
                    print("感谢使用中西医结合问诊系统，祝您健康！")
                    break
                else:
                    final_agent.reset_conversation()
                    print("新问诊开始，请描述您的症状...")
        
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。")
            print("\n" + "="*60)
            print("问诊过程总结")
            print("="*60)
            
            # 生成诊断总结
            if final_agent.conversation_history:
                summary = supervisor_agent.generate_final_summary(conversation_history)
                print(f"📋 问诊总结报告：")
                print(summary)
            else:
                print("没有问诊记录可以总结。")
            
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            print("请重试或联系系统管理员。")
            continue



if __name__ == "__main__":
    run_diagnosis_system()
