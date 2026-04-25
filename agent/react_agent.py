from langchain.agents import create_agent
from agent.tools.agent_tools import (rag_summarize, rerank_rag_search, query_rewrite_tool,
                                     compound_interest_calculator, loan_calculator, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from model.factory import chat_model
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, rerank_rag_search,query_rewrite_tool,compound_interest_calculator,loan_calculator, fill_context_for_report], #todo：修改工具
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1] #最后一条
            # 判断消息类型
            message_type = latest_message.type if hasattr(latest_message, 'type') else None

            # 如果是 AI 的最终回复（human/tool 类型的不展示）
            if latest_message.content and message_type == "ai":
                # 检查是否有 tool_calls，有则说明还在思考阶段
                if not hasattr(latest_message, 'tool_calls') or not latest_message.tool_calls:
                    # 这是最终答案，yield 出去
                    yield latest_message.content.strip() + "\n"
                else:
                    # 还在调用工具阶段，打印到控制台
                    logger.info(f"[Agent思考] {latest_message.content}")
            elif latest_message.content:
                # 其他类型消息（如工具返回结果），打印到控制台
                logger.info(f"[Agent总结] {latest_message.content[:200]}")
            # if latest_message.content:
            #     yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("手上有5万闲钱，应该怎么配置理财"):
        print(chunk, end="", flush=True)
