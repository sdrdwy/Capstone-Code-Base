#!/usr/bin/env python3

# 读取原文件
with open('/workspace/basic_app/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 新的异步函数定义
new_async_function = '''async def run_diagnosis_system_async():
    """异步版本的诊断系统"""
    print("="*60)
    print("欢迎使用中西医结合智能问诊系统 (异步版)")
    print("="*60)
    print("提示：输入 'quit' 或 'exit' 退出系统")
    print("-"*60)
    
    # 初始化组件
    llm, west_agent, tcm_agent, final_agent, supervisor_agent, tcm_vectorstore = initialize_components()
    
    try:
        while True:
            try:
                # 获取用户输入
                user_input = input("\\n患者: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("\\n感谢使用中西医结合问诊系统，祝您健康！")
                    break
                elif not user_input:
                    continue
                
                print("\\n正在分析您的症状...")
                
                # 让supervisor决定是否需要调用west_agent或tcm_agent
                conversation_history_str = "\\n".join(final_agent.conversation_history)
                agent_decision = supervisor_agent.decide_agent_usage(user_input, conversation_history_str)
                
                west_response = ""
                tcm_response = ""
                
                # 异步并行执行需要调用的查询
                tasks = []
                
                if agent_decision['should_call_west']:
                    print("🔍 正在进行西医知识检索...")
                    west_task = asyncio.create_task(
                        asyncio.get_event_loop().run_in_executor(
                            None, 
                            medical_qa_pipeline,
                            "qwen-max",
                            "./chroma_db_dash_w",
                            user_input
                        )
                    )
                    tasks.append(('west', west_task))
                
                if agent_decision['should_call_tcm']:
                    print("🌿 正在进行中医知识图谱查询...")
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
                    tasks.append(('tcm', tcm_task))
                
                # 等待查询任务完成并处理结果
                for agent_type, task in tasks:
                    try:
                        result = await task
                        if agent_type == 'west':
                            west_response = result['answer']
                        elif agent_type == 'tcm':
                            # 从tcm_result中提取结果，避免tcm_agent生成过长的总结
                            tcm_response = result.get('result', '') if isinstance(result, dict) else str(result)
                    except Exception as e:
                        print(f"⚠️ {agent_type}知识库查询出错: {str(e)}，使用空结果继续")
                        if agent_type == 'west':
                            west_response = ""
                        elif agent_type == 'tcm':
                            tcm_response = ""
                
                print("✅ 分析完成，正在整合信息...")
                
                # 交给final_agent处理
                final_response = final_agent.process_input(
                    patient_input=user_input,
                    west_response=west_response,
                    tcm_response=tcm_response
                )
                
                # 获取医生回复
                doctor_response = final_response['response']
                print(f"\\n👨‍⚕️ 医生: {doctor_response}")
                
                # supervisor_agent评估对话并决定是否提供建议
                conversation_history = "\\n".join(final_agent.conversation_history)
                supervision_result = supervisor_agent.evaluate_conversation(conversation_history)
                
                if supervision_result['should_advise'] and supervision_result['advice']:
                    print(f"\\n🎓 专家建议: {supervision_result['advice']}")
                
                # 检查是否结束对话
                if final_response['is_ended']:
                    print("\\n" + "="*60)
                    print("问诊结束")
                    print("="*60)
                    
                    # 生成诊断过程分析
                    analysis = final_agent.analyze_diagnosis_process()
                    print(f"\\n📋 问诊过程分析与建议：")
                    print(analysis)
                    
                    # 询问是否开始新对话
                    continue_diag = input("\\n是否开始新的问诊？(y/n): ").strip().lower()
                    if continue_diag not in ['y', 'yes', '是', 'Y']:
                        print("\\n感谢使用中西医结合问诊系统，祝您健康！")
                        break
                    else:
                        final_agent.reset_conversation()
                        print("\\n新问诊开始，请描述您的症状...")

            except KeyboardInterrupt:
                print("\\n\\n程序被用户中断。")
                # supervisor_agent自动总结问诊过程
                conversation_history = "\\n".join(final_agent.conversation_history)
                if conversation_history.strip():  # 如果有对话历史
                    print("🎓 正在生成问诊过程总结...")
                    summary = supervisor_agent.generate_summary(conversation_history)
                    print(f"\\n📋 问诊过程总结：")
                    print(summary)
                break
            except Exception as e:
                print(f"\\n❌ 发生错误: {str(e)}")
                print("请重试或联系系统管理员。")
                continue
    except KeyboardInterrupt:
        # 处理最外层的中断
        print("\\n\\n程序被用户中断。")
        conversation_history = "\\n".join(final_agent.conversation_history)
        if conversation_history.strip():  # 如果有对话历史
            print("🎓 正在生成问诊过程总结...")
            summary = supervisor_agent.generate_summary(conversation_history)
            print(f"\\n📋 问诊过程总结：")
            print(summary)'''

# 找到旧异步函数的起始和结束位置
start_marker = "async def run_diagnosis_system_async():"
end_marker = "            continue\n\n"

# 找到起始位置
start_pos = content.find(start_marker)
if start_pos == -1:
    print("未找到起始标记")
    exit(1)

# 找到结束位置（在起始位置之后）
end_pos = content.find(end_marker, start_pos)
if end_pos == -1:
    print("未找到结束标记")
    exit(1)

# 包含结束标记的完整结束位置
end_pos += len(end_marker)

# 替换内容
new_content = content[:start_pos] + new_async_function + content[end_pos:]

# 写回文件
with open('/workspace/basic_app/main.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("异步函数已更新成功！")