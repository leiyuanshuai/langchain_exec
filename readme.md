# 实现步骤

### 1: 先使用TextLoader加载本地知识库文档, 然后使用MarkdownHeaderTextSplitter分割器分隔Markdown

### 2: 把文档转换为向量存入Chroma向量数据库 并且返回vector_db实例

### 3: 每次用户的问题过来之后，先调用大模型再通过JsonOutputParser转换为{retrieve_flag, code_flag}类似的字典
#### 1: retrieve_flag为N 与文档知识库无关的问题,则直接透传question问题,构造提示词模版
```
[
 ("system", "你是一个智能助手"),
 ....history,
 ("human", "你是谁")
]

```
#### 2: 如果retrieve_flag为Y 则调用vector_db查询内容 根据code_flag=N 不是代码类问题,则使用
语义相似算法去寻找答案并且返回,否则使用最大余弦相似度算法寻找答案

### 4: 最后构造提示词模版
```

 ("system", "你是一个智能助手"),
 ....history, // 每次内容变化都追加到历史记录中
 ("human", "use_content")

use_content = f"""
  # 任务
  你需要参考如下内容来回答用户问题：
  # 参考内容
  {content} content是向量数据库搜索出来的答案
  # 问题
  {question}
  """
```



