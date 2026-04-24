
"""
发挥RAG作用：核心功能：搜索拼接、重排序、重写
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts,load_rewrite_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from utils.logger_handler import logger
from rag.rerank_service import rerank_service


def print_prompt(prompt): #调试用 把prompt打印到控制台
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_text_for_rewrite = load_rewrite_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    #搜索参考资料
    def rag_summarize(self, query: str, use_rerank: bool = False) -> str:
        logger.info(f"[retriever_docs] 普通搜索参考资料: {query}")

        context_docs = self.retriever_docs(query) #寻找搜索文档

        # 根据参数决定是否启用重排序
        if use_rerank and context_docs:
            context_docs = rerank_service.rerank_documents(query, context_docs)

        if not context_docs:
            logger.warning(f"[retriever_docs] 搜索参考资料无结果: {query}")
            return "抱歉，我在知识库中没有找到与您问题相关的信息。"

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            score = doc.metadata.get("rerank_score", "N/A")
            context += f"【参考资料{counter}】(相关性：{score}): 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )

    def query_rewrite_tool(self,query: str) -> str:
        """
        将用户的自然语言问题改写为更适合 RAG 检索的专业表述。
        """
        rewrite_prompt = PromptTemplate.from_template(self.prompt_text_for_rewrite)

        try:
            chain = rewrite_prompt | chat_model
            rewritten_query = chain.invoke({"query": query}).content.strip()
            logger.info(f"[query_rewrite_tool] 原始查询: {query} -> 改写后: {rewritten_query}")
            return rewritten_query
        except Exception as e:
            logger.error(f"[query_rewrite_tool] 改写出错: {e}")
            return query  # 出错时返回原查询，保证流程不中断

if __name__ == '__main__':
    rag = RagSummarizeService()

    # print(rag.rag_summarize("小户型适合哪些扫地机器人"))
    print(rag.query_rewrite_tool("手头有5万闲钱，怎么理财？"))
