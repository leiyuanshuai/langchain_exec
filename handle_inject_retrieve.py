from os.path import split

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter
from create_llm import  create_embeddings
file_path_list = [
    "./data/百炼_图像局部重绘.md",
    "./data/百炼_涂鸦作画.md",
]

def load_and_process_document():
   documents = []
   for file_path in file_path_list:
     loader = TextLoader(file_path=file_path, encoding="utf-8")
     row_documents = loader.load()
     document_content = row_documents[0].page_content
     splitter = MarkdownHeaderTextSplitter(
       headers_to_split_on=[
         ("#", "一级标题"),
         ("##", "二级标题"),
         ("###", "三级标题"),
       ],
       strip_headers=False, # 保留标题文本在分隔后的内容中
     )
     file_content_list = splitter.split_text(document_content)
     documents.extend(file_content_list)
   return documents

def create_vector_store(documents):
  embedding = create_embeddings(platform_code="local-embedding")
  vector_db = Chroma.from_documents(embedding=embedding,documents=documents)
  return vector_db  # 返回的是向量数据库实例

# load_and_process_document()
documents = load_and_process_document()
vector_db = create_vector_store(documents)
# print(vector_db)

def retrieve_from_vector(question: str, isCode=False):
    if isCode: # 如果是代码类问题 就使用最大边际相关性搜索算法获取相关文档 # k=1只是为了调试用
      docs = vector_db.max_marginal_relevance_search(query=question, k=1)
    else:
      # k=1只是为了调试用
      docs = vector_db.similarity_search(query=question, k=1)
    if len(docs) == 0:
        return  None
    # 构建包含检索结果的字符串列表内容
    doc_content = [doc.page_content for doc in docs]
    return "\n\n".join(doc_content)


def inject_retrieve_content(question: str, isCode=False):
  content = retrieve_from_vector(question, isCode)
  return f"""
  # 任务
  你需要参考如下内容来回答用户问题：
  # 参考内容
  {content}
  # 问题
  {question}
  """
# print(inject_retrieve_content("你是谁", False))

