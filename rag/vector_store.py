from langchain_milvus import Milvus
from langchain_core.documents import Document
from utils.config_handler import milvus_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader,md_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os
from pymilvus import connections, utility, Collection

#todo：修改为milvus原生适配写法 目前已可以读取上传的pdf和txt文件 pdf解析有待提高（暂缓实现）

class VectorStoreService:
    def __init__(self):
        connection_args = {
            "host": milvus_conf["host"],
            "port": milvus_conf["port"],
        }
        # self.connection_args = connection_args
        # self.collection_name = milvus_conf["collection_name"]
        #
        # # 确保集合存在且结构正确
        # self._ensure_collection()

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

    def _ensure_collection(self):
        """
        检查并创建/重建 Milvus 集合
        如果集合不存在则创建，如果存在但字段不匹配则重建
        """
        try:
            # 连接到 Milvus
            connections.connect(
                alias="default",
                host=self.connection_args["host"],
                port=self.connection_args["port"]
            )

            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                logger.info(f"[Milvus] 集合 '{self.collection_name}' 已存在")

                # # 检查集合结构是否兼容（可选）
                # collection = Collection(self.collection_name)
                # schema = collection.schema
                #
                # # 获取现有字段名
                # existing_fields = [field.name for field in schema.fields]
                # logger.info(f"[Milvus] 当前集合字段: {existing_fields}")
                #
                # # 如果缺少动态字段支持，建议重建
                # if not schema.enable_dynamic_field:
                #     logger.warning(f"[Milvus] 集合未启用动态字段，可能需要重建以支持元数据扩展")
            else:
                logger.info(f"[Milvus] 集合 '{self.collection_name}' 不存在，将在首次插入时自动创建")

        except Exception as e:
            logger.error(f"[Milvus] 检查集合状态失败: {e}")
            # 不阻断初始化，让 Milvus 在 add_documents 时自动创建

    def recreate_collection(self):
        """
        删除并重建集合（用于清空数据或修复结构问题）
        谨慎使用：会清除所有已有数据！
        """
        try:
            connections.connect(
                alias="default",
                host=self.connection_args["host"],
                port=self.connection_args["port"]
            )

            if utility.has_collection(self.collection_name):
                logger.warning(f"[Milvus] 正在删除旧集合 '{self.collection_name}'...")
                utility.drop_collection(self.collection_name)
                logger.info(f"[Milvus] 集合 '{self.collection_name}' 已删除")

            ##清空MD5
            if os.path.exists(get_abs_path(milvus_conf["md5_hex_store"])):
                os.remove(get_abs_path(milvus_conf["md5_hex_store"]))
            open(get_abs_path(milvus_conf["md5_hex_store"]), "w", encoding="utf-8").close()
            logger.info(f"[Milvus] 新集合将在首次插入数据时自动创建")

            # 重新初始化 vector_store
            self.vector_store = Milvus(
                embedding_function=embed_model,
                collection_name=self.collection_name,
                connection_args=self.connection_args,
                auto_id=True,
            )

            return True

        except Exception as e:
            logger.error(f"[Milvus] 重建集合失败: {e}")
            return False

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
            if read_path.endswith("txt") :
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            if read_path.endswith(".md"):
                return md_loader(read_path)

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
    vs.recreate_collection()
    vs.load_document() ## 需要手动load文档

    # retriever = vs.get_retriever()
    #
    # res = retriever.invoke("迷路")
    # for r in res:
    #     print(r.page_content)
    #     print("-" * 20)
