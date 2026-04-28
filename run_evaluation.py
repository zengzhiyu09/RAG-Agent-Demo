"""
RAGAS 评估工具
用于批量评估 RAG 系统性能
"""
from rag.rag_service import RagSummarizeService
from rag.evaluate_service import ragas_evaluator
from utils.logger_handler import logger
from agent.react_agent import ReactAgent
import json


def get_mock_agent_response(question: str) -> str:
    """
    模拟 Agent 回答（测试用）
    """
    mock_responses = {
        "手头有5万闲钱，应该怎么配置理财？":
            "建议您将5万元进行分散配置：\n\n1. **应急资金**（1-2万元）：存入货币基金或银行活期理财，保持流动性\n2. **稳健理财**（2-3万元）：购买银行定期理财产品、国债或大额存单，年化收益约2.5%-3%\n3. **进取投资**（剩余部分）：可考虑指数基金定投，长期持有获取市场平均收益\n\n**风险提示**：理财有风险，投资需谨慎。建议根据自身的风险承受能力合理配置，不要将所有资金投入单一产品。",

        "2026年银行定期存款利率是多少？":
            "2026年各大银行定期存款利率如下：\n\n- **一年期定存**：1.5%-2.0%\n- **三年期定存**：2.5%-3.0%\n- **五年期定存**：2.8%-3.3%\n\n具体利率因银行而异，国有大行利率相对较低，股份制银行和城商行可能略高。建议对比多家银行后选择。\n\n*注：以上为参考利率，实际以各银行官方公布为准*",

        "什么是复利计算？":
            "复利是指在计算利息时，不仅对本金计算利息，还对之前产生的利息继续计算利息，即'利滚利'。复利计算公式为：FV = PV × (1 + r)^n，其中FV是终值，PV是现值，r是利率，n是期数。",
        "帮我计算10万元存3年定期，年利率3%，复利最终能拿到多少钱？":
            "根据计算结果：\n\n- 您存入本金 **100,000元**  \n- 年利率为 **3%**，按复利计息  \n- 存期 **3年**  \n**3年后本息合计约为 109,272.70 元**，其中**利息为 9,272.70 元**。\n\n> 温馨提示：实际存款时，银行可能会根据政策调整利率，也可能按“利随本清”或“按年付息”等方式处理，具体以银行网点为准。",
        "我想贷款买房，贷款50万，期限20年，年利率4.5%，等额本息每月还多少？":
            "根据您的贷款条件（贷款50万元，期限20年，年利率4.5%，等额本息还款方式），每月需还款约 **3,163.25元**。\n整个贷款周期内，您总共需要支付利息约 **259,179.25元**，还款总额为 **759,179.25元**。\n温馨提示：\n - 等额本息的特点是每月还款额固定，便于规划预算；\n- 实际利率和还款计划可能因银行政策、征信情况等略有差异，建议办理前向具体银行确认最新政策。"}
    return mock_responses.get(question, "抱歉我无法回答问题")

def run_evaluation():
    """
    运行 RAGAS 评估
    """
    agent = ReactAgent()

    # 定义测试集（问题 + 标准答案）
    test_cases = [
        {
            "question": "手头有5万闲钱，应该怎么配置理财？",
            "ground_truth": "建议将5万元分为三部分：1. 保留1-2万元作为应急资金，存入货币基金或银行活期；2. 2-3万元购买稳健理财产品，如银行定期理财、国债等；3. 剩余部分可考虑指数基金定投。注意分散风险，不要全部投入高风险产品。"
        },
        {
            "question": "2026年银行定期存款利率是多少？",
            "ground_truth": "2026年各大银行定期存款利率有所不同，一般一年期定存利率在1.5%-2.0%之间，三年期在2.5%-3.0%之间，五年期在2.8%-3.3%之间。具体利率请以各银行官方公布为准。"
        },
        {
            "question": "什么是复利计算？",
            "ground_truth": "复利是指在计算利息时，不仅对本金计算利息，还对之前产生的利息继续计算利息，即'利滚利'。复利计算公式为：FV = PV × (1 + r)^n，其中FV是终值，PV是现值，r是利率，n是期数。"
        },
        {
            "question": "帮我计算10万元存3年定期，年利率3%，复利最终能拿到多少钱？",
            "ground_truth": "10万元存3年，年利率3%，按复利计算：终值 = 100000 × (1+0.03)^3 = 109272.7元，利息收益为9272.7元。"
        },
        {
            "question": "我想贷款买房，贷款50万，期限20年，年利率4.5%，等额本息每月还多少？",
            "ground_truth": "贷款50万，20年（240期），年利率4.5%（月利率0.375%），等额本息月供约为3163.25元，总还款额约75.92万元，总利息约25.92万元。"
        },
    ]

    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    logger.info("[RAGAS评估] 开始收集Agent回答...")

    for idx, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        ground_truth = test_case.get("ground_truth")

        logger.info(f"[RAGAS评估] [{idx}/{len(test_cases)}] 处理问题: {question}")

        # 获取 Agent 的最终回答
        full_response = ""
        try:
            for chunk in agent.execute_stream(question):
                full_response += chunk
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            logger.error(f"[RAGAS评估] Agent 执行失败: {e}")
            full_response = f"执行出错: {str(e)}"

        questions.append(question)
        answers.append(full_response)
        ground_truths.append(ground_truth)

        logger.info(f"[RAGAS评估] 回答长度: {len(full_response)} 字符")
        # mock_answer = get_mock_agent_response(question)
        #
        # questions.append(question)
        # answers.append(mock_answer)
        # ground_truths.append(ground_truth)
    logger.info("[RAGAS评估] 开始执行 RAGAS 评估...")

    # 执行批量评估（不提供 contexts，只评估最终答案）
    df = ragas_evaluator.evaluate_batch(
        questions=questions,
        answers=answers,
        contexts_list=[[] for _ in questions],
        ground_truths=ground_truths
    )
    #输出报告：可选也可不选
    if not df.empty:
        # 生成报告
        report = ragas_evaluator.get_evaluation_report(df)
        print("\n" + "=" * 60)
        print(report)
        print("=" * 60)

        logger.info("[RAGAS评估] 评估完成！结果已保存到 logs 目录")
    else:
        logger.error("[RAGAS评估] 评估失败，未生成结果")


if __name__ == "__main__":
    run_evaluation()