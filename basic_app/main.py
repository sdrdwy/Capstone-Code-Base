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

import asyncio
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_neo4j import Neo4jGraph

from .agents.west_agent import WestAgent, medical_qa_pipeline
from .agents.tcm_agent import TcmAgent
from .agents.supervisor_agent import SupervisorAgent
from .final_agent import FinalAgent
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
    
    # 初始化各Agent
    west_agent = WestAgent(
        llm=llm,
        retriever=west_vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    
    tcm_agent = TcmAgent(
        llm=llm,
        graph=graph
    )
    
    final_agent = FinalAgent(
        llm=llm,
        west_agent=west_agent,
        tcm_agent=tcm_agent
    )
    
    supervisor_agent = SupervisorAgent(llm=llm)
    
    print("系统初始化完成！\n")
    
    return llm, west_agent, tcm_agent, final_agent, supervisor_agent, tcm_vectorstore


def should_call_agents(user_input, conversation_history, llm):
    """决定是否调用west_agent或tcm_agent"""
    # 使用LLM来判断是否需要调用西医或中医agent
    decision_prompt = f"""
    作为中西医结合专家，请判断用户的问题是否需要调用西医知识库、中医知识库或两者都需要。
    
    用户输入：{user_input}
    对话历史：{conversation_history}
    
    请返回一个JSON格式的结果，包含以下字段：
    {{
      "call_west": true/false,
      "call_tcm": true/false
    }}
    
    如果用户问题明确涉及西医术语或现代医学概念，返回call_west为true。
    如果用户问题涉及中医术语、证型、方剂等，返回call_tcm为true。
    如果问题涉及两者，返回两者都为true。
    如果问题不涉及医学知识，返回两者都为false。
    """
    
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import json
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", decision_prompt)
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({})
    
    try:
        # 尝试解析JSON响应
        result = json.loads(response)
        return result.get("call_west", True), result.get("call_tcm", True)
    except:
        # 如果解析失败，返回默认值
        return True, True


def run_diagnosis_system():
    """运行诊断系统主循环"""
    print("="*60)
    print("欢迎使用中西医结合智能问诊系统")
    print("="*60)
    print("提示：输入 'quit' 或 'exit' 退出系统")
    print("输入 'reset' 重置对话")
    print("-"*60)
    
    # 初始化组件
    llm, west_agent, tcm_agent, final_agent, supervisor_agent, tcm_vectorstore = initialize_components()
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n患者: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n感谢使用中西医结合问诊系统，祝您健康！")
                
                # 在退出前进行总结
                if final_agent.conversation_history:
                    print("\n" + "="*60)
                    print("问诊过程总结")
                    print("="*60)
                    analysis = final_agent.analyze_diagnosis_process()
                    print(f"\n📋 问诊过程分析与建议：")
                    print(analysis)
                break
            elif user_input.lower() == 'reset':
                final_agent.reset_conversation()
                print("对话已重置。")
                continue
            elif not user_input:
                continue
            
            print("\n正在分析您的症状...")
            
            # 决定是否调用west_agent或tcm_agent
            call_west, call_tcm = should_call_agents(user_input, 
                                                   "\n".join(final_agent.conversation_history), 
                                                   llm)
            
            # 并行执行西医和中医查询（根据需要）
            west_response = "无相关信息"
            tcm_response = "无相关信息"
            
            if call_west:
                print("🔍 正在进行西医知识检索...")
                try:
                    west_result = medical_qa_pipeline(
                        llm_choice="qwen-max",
                        vector_db_path="./chroma_db_dash_w",
                        user_query=user_input
                    )
                    west_response = west_result['answer']
                except Exception as e:
                    print(f"⚠️ 西医查询出现问题: {str(e)}，继续运行...")
                    west_response = "西医知识库查询失败，无相关信息"
            
            if call_tcm:
                print("🌿 正在进行中医知识图谱查询...")
                try:
                    # 首先修复查询
                    fixed_query_result = fix_query(user_input, llm, tcm_vectorstore, 10)
                    fixed_query = fixed_query_result['query']
                    # 然后进行图谱查询
                    tcm_result = tcm_agent.query(fixed_query)
                    tcm_response = tcm_result.get('result', '中医知识库查询失败，无相关信息')
                except Exception as e:
                    print(f"⚠️ 中医查询出现问题: {str(e)}，继续运行...")
                    tcm_response = "中医知识库查询失败，无相关信息"
            
            print("✅ 分析完成，正在整合信息...")
            
            # 交给final_agent处理
            final_response = final_agent.process_input(
                patient_input=user_input,
                west_response=west_response,
                tcm_response=tcm_response
            )
            
            # 获取医生回复
            doctor_response = final_response['response']
            print(f"\n👨‍⚕️ 医生: {doctor_response}")
            
            # supervisor_agent评估对话并决定是否提供建议
            conversation_history = "\n".join(final_agent.conversation_history)
            supervision_result = supervisor_agent.evaluate_conversation(conversation_history, west_agent, tcm_agent, user_input)
            
            if supervision_result['should_advise'] and supervision_result['advice']:
                print(f"\n🎓 专家建议: {supervision_result['advice']}")
            
            # 检查是否结束对话
            if final_response['is_ended']:
                print("\n" + "="*60)
                print("问诊结束")
                print("="*60)
                
                # 生成诊断过程分析
                analysis = final_agent.analyze_diagnosis_process()
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
            print("KBInterrupt触发 - 问诊过程总结")
            print("="*60)
            
            # KBInterrupt后自动总结问诊过程
            if final_agent.conversation_history:
                analysis = final_agent.analyze_diagnosis_process()
                print(f"\n📋 问诊过程分析与建议：")
                print(analysis)
            else:
                print("当前没有问诊记录。")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            print("请重试或联系系统管理员。")
            continue


async def run_diagnosis_system_async():
    """异步版本的诊断系统"""
    print("="*60)
    print("欢迎使用中西医结合智能问诊系统 (异步版)")
    print("="*60)
    print("提示：输入 'quit' 或 'exit' 退出系统")
    print("-"*60)
    
    # 初始化组件
    llm, west_agent, tcm_agent, final_agent, supervisor_agent, tcm_vectorstore = initialize_components()
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n患者: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("\n感谢使用中西医结合问诊系统，祝您健康！")
                
                # 在退出前进行总结
                if final_agent.conversation_history:
                    print("\n" + "="*60)
                    print("问诊过程总结")
                    print("="*60)
                    analysis = final_agent.analyze_diagnosis_process()
                    print(f"\n📋 问诊过程分析与建议：")
                    print(analysis)
                break
            elif not user_input:
                continue
            
            print("\n正在分析您的症状...")
            
            # 决定是否调用west_agent或tcm_agent
            call_west, call_tcm = should_call_agents(user_input, 
                                                   "\n".join(final_agent.conversation_history), 
                                                   llm)
            
            # 异步并行执行西医和中医查询（根据需要）
            west_response = "无相关信息"
            tcm_response = "无相关信息"
            
            if call_west:
                print("🔍 正在进行西医知识检索...")
                try:
                    west_task = asyncio.create_task(
                        asyncio.get_event_loop().run_in_executor(
                            None, 
                            medical_qa_pipeline,
                            "qwen-max",
                            "./chroma_db_dash_w",
                            user_input
                        )
                    )
                except Exception as e:
                    print(f"⚠️ 西医查询出现问题: {str(e)}，继续运行...")
                    west_response = "西医知识库查询失败，无相关信息"
            else:
                west_task = None
            
            if call_tcm:
                print("🌿 正在进行中医知识图谱查询...")
                try:
                    # 修复查询
                    fixed_query_task = asyncio.create_task(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            fix_query,
                            user_input,
                            llm,
                            tcm_vectorstore,
                            10
                        )
                    )

                    # 等待查询修复完成
                    fixed_query_result = await fixed_query_task
                    fixed_query = fixed_query_result['query']

                    # 进行图谱查询
                    tcm_task = asyncio.create_task(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            tcm_agent.query,
                            fixed_query
                        )
                    )
                    
                    # 等待查询完成
                    if west_task:
                        west_result, tcm_result = await asyncio.gather(west_task, tcm_task, return_exceptions=True)
                        if isinstance(west_result, Exception):
                            print(f"⚠️ 西医查询出现问题: {str(west_result)}，继续运行...")
                            west_response = "西医知识库查询失败，无相关信息"
                        else:
                            west_response = west_result['answer']
                        
                        if isinstance(tcm_result, Exception):
                            print(f"⚠️ 中医查询出现问题: {str(tcm_result)}，继续运行...")
                            tcm_response = "中医知识库查询失败，无相关信息"
                        else:
                            tcm_response = tcm_result.get('result', '中医知识库查询失败，无相关信息')
                    else:
                        tcm_result = await tcm_task
                        if isinstance(tcm_result, Exception):
                            print(f"⚠️ 中医查询出现问题: {str(tcm_result)}，继续运行...")
                            tcm_response = "中医知识库查询失败，无相关信息"
                        else:
                            tcm_response = tcm_result.get('result', '中医知识库查询失败，无相关信息')
                except Exception as e:
                    print(f"⚠️ 中医查询出现问题: {str(e)}，继续运行...")
                    tcm_response = "中医知识库查询失败，无相关信息"
            else:
                if west_task:
                    west_result = await west_task
                    if isinstance(west_result, Exception):
                        print(f"⚠️ 西医查询出现问题: {str(west_result)}，继续运行...")
                        west_response = "西医知识库查询失败，无相关信息"
                    else:
                        west_response = west_result['answer']
            
            print("✅ 分析完成，正在整合信息...")
            
            # 交给final_agent处理
            final_response = final_agent.process_input(
                patient_input=user_input,
                west_response=west_response,
                tcm_response=tcm_response
            )
            
            # 获取医生回复
            doctor_response = final_response['response']
            print(f"\n👨‍⚕️ 医生: {doctor_response}")
            
            # supervisor_agent评估对话并决定是否提供建议
            conversation_history = "\n".join(final_agent.conversation_history)
            supervision_result = supervisor_agent.evaluate_conversation(conversation_history, west_agent, tcm_agent, user_input)
            
            if supervision_result['should_advise'] and supervision_result['advice']:
                print(f"\n🎓 专家建议: {supervision_result['advice']}")
            
            # 检查是否结束对话
            if final_response['is_ended']:
                print("\n" + "="*60)
                print("问诊结束")
                print("="*60)
                
                # 生成诊断过程分析
                analysis = final_agent.analyze_diagnosis_process()
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
            print("KBInterrupt触发 - 问诊过程总结")
            print("="*60)
            
            # KBInterrupt后自动总结问诊过程
            if final_agent.conversation_history:
                analysis = final_agent.analyze_diagnosis_process()
                print(f"\n📋 问诊过程分析与建议：")
                print(analysis)
            else:
                print("当前没有问诊记录。")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            print("请重试或联系系统管理员。")
            continue


if __name__ == "__main__":
    # 选择运行同步或异步版本
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--async":
        asyncio.run(run_diagnosis_system_async())
    else:
        run_diagnosis_system()