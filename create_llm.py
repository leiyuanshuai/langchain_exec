import json

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from ai_configs import default_ai_configs
from langchain_openai import ChatOpenAI


def create_llm(platform_code="local-embedding", temperature=0):
    _ai_config = default_ai_configs.get(platform_code)

    if _ai_config is None:
        raise Exception("Unknown platform code", platform_code)

    return ChatOpenAI(
        base_url=_ai_config.get("url").replace("chat/completions", ""),
        api_key=_ai_config.get("key"),
        model=_ai_config.get("model"),
        temperature=temperature,
        seed=2,  # 同样的seed每次生成的结果相同
    )


def create_embeddings(platform_code="huoshan-embedding-240715"):
    """
    创建自定义嵌入模型实例

    参数:
    platform_code: 平台代码，用于从默认配置中查找对应平台的API信息

    返回:
    CustomEmbeddings类的实例，用于生成文本嵌入向量

    异常:
    当找不到对应平台代码的配置时抛出异常
    """
    # 从默认配置中获取指定平台的AI配置信息
    _ai_config = default_ai_configs.get(platform_code)

    # 检查配置是否存在
    if _ai_config is None:
        raise Exception("Unknown platform code", platform_code)

    # 创建并返回自定义嵌入模型实例
    return CustomEmbeddings(
        base_url=_ai_config.get("url").replace("/embeddings", ""),  # API基础URL
        api_key=_ai_config.get("key"),  # API密钥
        model=_ai_config.get("model"),  # 嵌入模型名称
    )


class CustomEmbeddings(Embeddings):
    """
    自定义文本嵌入类，用于将文本转换为向量表示
    继承自LangChain的Embeddings基类
    """

    def __init__(self, base_url, api_key, model):
        """
        初始化自定义嵌入类

        参数:
        base_url: API的基础URL
        api_key: 访问API所需的密钥
        model: 要使用的嵌入模型名称
        """
        self.base_url = base_url  # API基础URL
        self.api_key = api_key  # API访问密钥
        self.model = model  # 嵌入模型名称

    def embed_documents(self, texts):
        """
        将多个文档转换为嵌入向量

        参数:
        texts: 包含多个文本的列表

        返回:
        包含每个文本对应嵌入向量的列表
        """
        # 设置请求头，包括内容类型和认证信息
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # 构建请求负载
        payload = {"input": texts, "model": self.model, "encoding_format": "float"}

        # 发送POST请求到嵌入API
        response = requests.post(
            f"{self.base_url}/embeddings", headers=headers, data=json.dumps(payload)
        )

        # 检查请求是否成功，如果失败则抛出异常
        response.raise_for_status()

        # 解析响应JSON数据
        json_data = response.json()

        # 从响应数据中提取嵌入向量并返回
        return [item["embedding"] for item in json_data["data"]]

    def embed_query(self, text):
        """
        将单个查询文本转换为嵌入向量

        参数:
        text: 查询文本

        返回:
        对应的嵌入向量
        """
        # 调用embed_documents处理单个文本，并返回第一个结果
        return self.embed_documents([text])[0]


def chain_log(val):
    print("chain log:", val, type(val))
    return val


runnable_chain_log = RunnableLambda(chain_log)


def specific_chain_log(format_func):
    def func(val):
        print("specific chain log:", format_func(val))
        return val

    return func


def runnable_specific_chain_log(format_func):
    def func(val):
        print("runnable specific chain log:", format_func(val))
        return val

    return RunnableLambda(func)
