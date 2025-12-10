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
    embedding_v3 = DashScopeEmbeddings(model = "text-embedding-v3")
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

    # 初始化各Agent
    west_agent = WestAgent(
        llm=llm,
        retriever=west_vectorstore.as_retriever()
    )
    
    tcm_agent = TcmAgent(
        llm=llm,
        graph=graph,
        retriever=tcm_med_vectorstore.as_retriever()
    )
    
    final_agent = FinalAgent(
        llm=llm,
        # west_agent=west_agent,
        # tcm_agent=tcm_agent
    )
    
    supervisor_agent = SupervisorAgent(llm=llm)
    
    print("系统初始化完成！\n")
    
    return llm, west_agent, tcm_agent, final_agent, supervisor_agent, tcm_vectorstore


def run_diagnosis_system(enable_advice = True):
    """运行诊断系统主循环"""
    print("="*60)
    print("欢迎使用中西医结合智能问诊系统")
    print("="*60)
    print("提示：输入 'quit' 或 'exit' 退出系统")
    print("输入 'reset' 重置对话")
    print("-"*60)
    
    # 初始化组件
    llm, west_agent, tcm_agent, final_agent, supervisor_agent, tcm_vectorstore = initialize_components()
    supervision_advice = ""
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
            
            # 交给final_agent处理
            final_response = final_agent.process_input(
                patient_input=user_input,
                advice=supervision_advice
            )
            
            # 获取医生回复
            doctor_response = final_response['response']
            print(f"\n👨‍⚕️ 医生: {doctor_response}")
            
            # supervisor_agent评估对话并决定是否提供建议
            conversation_history = final_agent.conversation_history


            fixed_query = fix_query(user_input,llm,tcm_vectorstore,10)['query']

            west_response = west_agent.query(user_input,conversation_history)
            tcm_response = tcm_agent.query(fixed_query,conversation_history)

            supervision_result = supervisor_agent.evaluate_conversation(conversation_history,
                                                                        tcm_response['result'],
                                                                        west_response['result'])
            
            if enable_advice == True:
                supervision_advice = supervision_result
            else: False
            if supervision_result['should_advise'] and supervision_result['advice']:
                print(f"\n🎓 专家建议: {supervision_result['advice']}")
            
            # 检查是否结束对话
            if final_response['is_ended']:
                print("\n" + "="*60)
                print("问诊结束")
                print("="*60)
                
                # 生成诊断过程分析
                analysis = supervisor_agent.analyze_diagnosis_process(final_agent.conversation_history)
                print(f"\n📋 问诊过程分析与建议：")
                print(analysis)
                
                # 询问是否开始新对话
                continue_diag = input("\n是否开始新的问诊？(y/n): ").strip().lower()
                if continue_diag not in ['y', 'yes', '是', 'Y']:
                    print("\n感谢使用中西医结合问诊系统，祝您健康！")
                    break
                else:
                    final_agent.reset_conversation()
                    print("\n新问诊开始，请描述您的症状...")
        
        except KeyboardInterrupt:
            print("\n\n程序被用户中断。")
            print("\n" + "="*60)
            print("问诊过程总结")
            print("="*60)
            
            # 生成诊断过程分析
            if final_agent.conversation_history:
                analysis = supervisor_agent.analyze_diagnosis_process(final_agent.conversation_history)
                print(f"\n📋 问诊过程分析与建议：")
                print(analysis)
            else:
                print("没有问诊记录可以总结。")
            
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            print("请重试或联系系统管理员。")
            continue



if __name__ == "__main__":
    import sys
    if "--disable" in sys.argv:
        run_diagnosis_system(enable_advice=True)
    else:
        run_diagnosis_system(enable_advice=False)