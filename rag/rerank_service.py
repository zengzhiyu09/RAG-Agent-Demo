from typing import List, Dict

from dashscope import TextReRank
from langchain_core.documents  import Document
from utils.logger_handler import logger


class RerankService:
    def __init__(self, model: str = "qwen3-rerank", top_n: int = 3):
        self.model = model
        self.top_n = top_n
        # 针对银行理财场景的指令引导
        self.instruction = (
            "You are a banking expert. Given a search query about finance, retrieve relevant passages that answer the query"
            
            "When the user mentions 'safe', 'stable','稳健' or 'low risk', prioritize documents explicitly stating 'principal guaranteed' (保本、低风险)."
        )


    def rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        """
        对召回文档进行重排序
        :param query: 用户查询
        :param docs: LangChain Document 对象列表
        :return: 重排序后的 Document 列表
        """
        if not docs:
            return []

        doc_texts = [doc.page_content for doc in docs]

        try:
            response = TextReRank.call(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=self.top_n,
                instruction=self.instruction,
                return_documents=False
            )

            if response.status_code != 200:
                raise Exception(f"Rerank API Error: {response.code} - {response.message}")

            results = []
            if response.output and response.output.results:
                for item in response.output.results:
                    results.append({"index": item.index, "score": item.relevance_score})

            # 根据索引重组 Document 对象
            sorted_docs = []
            for item in results:
                idx = item["index"]
                if 0 <= idx < len(docs):
                    docs[idx].metadata["rerank_score"] = item["score"]
                    sorted_docs.append(docs[idx])
            return sorted_docs

        except Exception as e:
            logger.error(f"[RerankService] 重排序失败，返回原始结果: {e}")
            return docs[:self.top_n]


# 全局单例
rerank_service = RerankService()


# ... existing code ...

if __name__ == "__main__":


    # 2. 模拟用户查询（带有明确的低风险偏好）
    user_query = "我手头有5万块闲钱，想找个绝对安全、保本的理财方式，不想承担任何风险"

    # 3. 模拟向量检索召回的原始文档（顺序是杂乱的，且包含干扰项）
    raw_docs = [
        Document(page_content="股票型基金：主要投资于二级市场股票，风险等级 R4，历史波动较大，适合长期投资。", metadata={"source": "fund_guide.txt"}),
        Document(page_content="大额存单：三年期利率 2.6%，受存款保险制度保护，50万以内绝对保本，安全性极高。", metadata={"source": "deposit_products.txt"}),
        Document(page_content="结构性存款：挂钩黄金或汇率，预期收益率 1.5%-3.5%，非保本浮动收益，存在本金损失可能。", metadata={"source": "structured_deposit.txt"}),
        Document(page_content="国债逆回购：短期理财工具，以国债为抵押，风险极低，收益稳定，适合闲置资金打理。", metadata={"source": "bond_reverse_repo.txt"}),
        Document(page_content="R5级激进型理财产品：主要投资于衍生品市场，可能产生巨额亏损，仅适合专业投资者。", metadata={"source": "high_risk_warning.txt"})
    ]

    print(f"🔍 用户查询: {user_query}")
    print(f"📚 原始召回数量: {len(raw_docs)}")
    print("-" * 50)

    # 4. 执行重排序
    sorted_docs = rerank_service.rerank_documents(user_query, raw_docs)

    # 5. 输出结果
    print("✅ 重排序后的结果 (Top 3):")
    for i, doc in enumerate(sorted_docs, 1):
        score = doc.metadata.get("rerank_score", 0)
        source = doc.metadata.get("source", "Unknown")
        print(f"Top {i} [分数: {score:.4f}] [来源: {source}]")
        print(f"   内容: {doc.page_content[:60]}...")
        print()

'''
✅ 重排序后的结果 (Top 3):
Top 1 [分数: 0.8109] [来源: bond_reverse_repo.txt]
   内容: 国债逆回购：短期理财工具，以国债为抵押，风险极低，收益稳定，适合**闲置资金**打理。...  好吧因为是闲钱。

Top 2 [分数: 0.7958] [来源: deposit_products.txt]
   内容: 大额存单：三年期利率 2.6%，受存款保险制度保护，50万以内绝对保本，安全性极高。...

Top 3 [分数: 0.4936] [来源: structured_deposit.txt]
   内容: 结构性存款：挂钩黄金或汇率，预期收益率 1.5%-3.5%，非保本浮动收益，存在本金损失可能。...


'''