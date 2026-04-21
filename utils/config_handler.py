"""
yaml
k: v
"""
import yaml
from utils.path_tool import get_abs_path
import os

# def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
#     with open(config_path, "r", encoding=encoding) as f:
#         return yaml.load(f, Loader=yaml.FullLoader)
#
#
# def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
#     with open(config_path, "r", encoding=encoding) as f:
#         return yaml.load(f, Loader=yaml.FullLoader)
#
#
# def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
#     with open(config_path, "r", encoding=encoding) as f:
#         return yaml.load(f, Loader=yaml.FullLoader)
#
#
# def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
#     with open(config_path, "r", encoding=encoding) as f:
#         return yaml.load(f, Loader=yaml.FullLoader)

def load_config(filename: str) -> dict:
    config_path = get_abs_path(os.path.join("config", filename))
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

rag_conf = load_config("rag.yml")
# chroma_conf = load_config("chroma.yml") deleted
prompts_conf = load_config("prompts.yml")
agent_conf = load_config("agent.yml")
milvus_conf = load_config("milvus.yml")


if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
