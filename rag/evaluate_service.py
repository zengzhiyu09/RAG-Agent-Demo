"""
RAGAS 评估服务
用于评估 RAG 系统的检索质量、生成质量和整体性能
"""
from ragas import evaluate
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness,
)
from datasets import Dataset
from typing import List, Dict, Optional
import pandas as pd

from model.factory import chat_model, embed_model
from utils.logger_handler import logger
from datetime import datetime


class RagasEvaluationService:
    def __init__(self, use_context_metrics: bool = True):
        """
        初始化评估服务
        Args:
            use_context_metrics: 是否使用依赖上下文的指标
                                 True: 包含所有指标（适合评估 RAG 检索质量）
                                 False: 仅使用答案相关指标（适合评估 Agent 最终输出）
        """
        self.llm = LangchainLLMWrapper(chat_model) #此处配置使用的也是同一个语言模型，小心token消耗！！
        self.embeddings = LangchainEmbeddingsWrapper(embed_model)
        if use_context_metrics:
            self.metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            ]
        else:
            self.metrics = [
                answer_relevancy,
                answer_correctness,
                answer_similarity,
            ]

        self.use_context_metrics = use_context_metrics

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        评估单个问答对
        """
        try:
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts if contexts else []],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)

            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )

            df = result.to_pandas()
            score_cols = [col for col in df.columns
                         if col not in ['question', 'answer', 'contexts', 'ground_truth']]
            scores = df[score_cols].iloc[0].to_dict()

            logger.info(f"[RAGAS评估] 问题: {question[:50]}...")
            logger.info(f"[RAGAS评估] 得分: {scores}")
            return scores

        except Exception as e:
            logger.error(f"[RAGAS评估] 评估失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        批量评估多个问答对
        """
        if ground_truths is None:
            ground_truths = [None] * len(questions)

        try:
            data = {
                "question": questions,
                "answer": answers,
                "contexts": [ctx if ctx else [] for ctx in contexts_list],
            }

            if ground_truths and any(gt is not None for gt in ground_truths):
                data["ground_truth"] = [gt if gt is not None else "" for gt in ground_truths]

            dataset = Dataset.from_dict(data)

            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,

            )

            df = result.to_pandas()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/ragas_eval_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')

            logger.info(f"[RAGAS评估] 批量评估完成，结果已保存到: {filename}")

            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                logger.info(f"[RAGAS评估] 平均得分:\n{df[numeric_cols].mean()}")

            return df

        except Exception as e:
            logger.error(f"[RAGAS评估] 批量评估失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def get_evaluation_report(
        self,
        df: pd.DataFrame
    ) -> str:
        """
        生成评估报告
        """
        if df.empty:
            return "评估结果为空"

        report = "=== RAGAS 评估报告 ===\n\n"
        report += f"评估样本数: {len(df)}\n\n"

        report += "各指标平均分:\n"
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            avg_score = df[col].mean()
            report += f"  - {col}: {avg_score:.4f}\n"

        report += "\n详细统计:\n"
        report += df[numeric_cols].describe().to_string()

        return report


# 全局实例 - 默认不使用上下文指标（适合评估 Agent 最终输出）
ragas_evaluator = RagasEvaluationService(use_context_metrics=False)
