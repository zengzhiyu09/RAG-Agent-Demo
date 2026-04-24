from langchain_milvus import Milvus
from langchain_core.documents import Document
from utils.config_handler import milvus_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os

#todo：修改为milvus原生适配写法 目前已可以读取上传的pdf和txt文件 pdf解析有待提高（暂缓实现）

class VectorStoreService:
    def __init__(self):
        connection_args = {
            "host": milvus_conf["host"],
            "port": milvus_conf["port"],
        }

        self.vector_store = Milvus(
            embedding_function=embed_model,
            collection_name=milvus_conf["collection_name"],
            connection_args=connection_args,
            auto_id=True,
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=milvus_conf["chunk_size"],
            chunk_overlap=milvus_conf["chunk_overlap"],
            separators=milvus_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": milvus_conf["k"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库
        计算文件的MD5做去重
        :return: None
        """

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(milvus_conf["md5_hex_store"])):
                open(get_abs_path(milvus_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False

            with open(get_abs_path(milvus_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True

                return False

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(milvus_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")





        ##只能读取txt和pdf

        def get_file_documents(read_path: str):
            if read_path.endswith("txt") or read_path.endswith("json"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)

            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(milvus_conf["data_path"]),
            tuple(milvus_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                self.vector_store.add_documents(split_document)

                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document() ## 需要手动load文档

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
