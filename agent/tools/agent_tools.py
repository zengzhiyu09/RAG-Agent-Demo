import os
import math
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

rag = RagSummarizeService()




@tool(description="从向量存储中检索参考资料，简单查询")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query,use_rerank=False)

@tool(description="带重排序检索。入参：query(检索词), use_rerank(是否开启重排序, 默认为True)。返回：经过重排序和总结的专业解答。")
def rerank_rag_search(query: str, use_rerank: bool = True) -> str:
    """
    带有可选重排序功能的 RAG 检索工具
    """
    return rag.rag_summarize(query, use_rerank=True)


@tool(
    description="查询改写工具。当用户的问题过于口语化、模糊或缺少关键金融术语时，先调用此工具将其优化为适合向量检索的专业查询语句。入参：query(原始问题)。返回：优化后的查询字符串。")
def query_rewrite_tool(query: str) -> str:
    """
    将用户的自然语言问题改写为更适合 RAG 检索的专业表述。
    """
    return rag.query_rewrite_tool(query)

@tool(
    description="计算复利或定期存款收益。入参：principal(本金), rate(年利率百分比, 如2.6表示2.6%), years(年限)。返回最终本息合计及利息总额。")
def compound_interest_calculator(principal: float, rate: float, years: int) -> str:
    """
    计算逻辑：A = P * (1 + r/n)^(nt)
    这里简化为按年复利或单利（根据银行常规，定期存款通常为单利，但理财产品可能涉及复利）
    此处采用通用复利公式，若需单利可调整为 P * (1 + r * t) #todo：单利复利转换
    """
    try:
        # 将百分比转换为小数
        r = rate / 100.0
        # 计算本息合计 (按年复利)
        amount = principal * math.pow((1 + r), years)
        interest = amount - principal
        return f"本金 {principal} 元，在年利率 {rate}% 的情况下，{years} 年后的本息合计约为 {amount:.2f} 元，其中利息为 {interest:.2f} 元。"
    except Exception as e:
        logger.error(f"[compound_interest_calculator] 计算出错: {e}")
        return "计算过程中出现错误，请检查输入参数。"


@tool(
    description="计算贷款月供。入参：principal(贷款本金), rate(年利率百分比), years(贷款年限), type(还款方式: 'equal_principal'等额本金 或 'equal_installment'等额本息)。返回月供信息。")
def loan_calculator(principal: float, rate: float, years: int, type: str = "equal_installment") -> str:
    """
    计算贷款月供
    """
    try:
        months = years * 12
        monthly_rate = (rate / 100.0) / 12 #单月的利率

        if type == "equal_installment":
            # 等额本息: [本金×月利率×(1+月利率)^还款月数]÷[(1+月利率)^还款月数-1]
            if monthly_rate == 0: #无息？
                monthly_payment = principal / months
            else:
                monthly_payment = (principal * monthly_rate * math.pow(1 + monthly_rate, months)) / (
                            math.pow(1 + monthly_rate, months) - 1)
            total_payment = monthly_payment * months
            total_interest = total_payment - principal
            return f"采用等额本息方式，每月月供约为 {monthly_payment:.2f} 元，总利息约为 {total_interest:.2f} 元。"

        elif type == "equal_principal":
            # 等额本金: 每月归还固定本金 + 剩余本金产生的利息
            monthly_principal = principal / months
            first_month_interest = principal * monthly_rate
            first_month_payment = monthly_principal + first_month_interest
            last_month_interest = monthly_principal * monthly_rate
            last_month_payment = monthly_principal + last_month_interest
            total_interest = (months + 1) * principal * monthly_rate / 2
            return f"采用等额本金方式，首月月供约为 {first_month_payment:.2f} 元，末月月供约为 {last_month_payment:.2f} 元，总利息约为 {total_interest:.2f} 元。"

        else:
            return "不支持的还款方式，请选择 'equal_installment' (等额本息) 或 'equal_principal' (等额本金)。"

    except Exception as e:
        logger.error(f"[loan_calculator] 计算出错: {e}")
        return "计算过程中出现错误，请检查输入参数。"




@tool(description="无入参，无返回值，调用后触发中间件自动为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "fill_context_for_report已调用"
