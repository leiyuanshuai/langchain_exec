from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel,Field
from create_history_manage import create_history_manage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from create_llm import create_llm
save_message,get_history_list, print_history_list = create_history_manage()
from handle_inject_retrieve import inject_retrieve_content, file_path_list


class RetrieveInfo(BaseModel):
      retrieve_flag: str = Field(description="是否需要检索文档信息, 值为Y或N")
      code_flag: str = Field(description="是否为代码类的问题，值为Y或N")


format_retrieve_result = ChatPromptTemplate.from_template(
      """
# 任务
请判断用户的问题是否存在与文档主题存在相同的名词:
返回的结果必须是一个json对象，对象有retrieve_flag和code_flag两个属性,两个属性的值都是只有Y或N两个取值;
存在相同名词，输出结果的 retrieve_flag 为Y，否则为N，
在存在相同名词的基础上，如果问题类型为代码问题，则输出结果的 code_flag 为Y，否则为N
如果不存在相同名词，则输出结果的 retrieve_flag 以及 code_flag 都是N

# 文档主题
{file_list}
# 问题
{question}

""") | create_llm("local") | JsonOutputParser(pydantic_object=RetrieveInfo)

# print(format_retrieve_result.invoke({
#       "file_list": ",".join(file_path_list),
#       "question": "你是谁",
# }))

# print(format_retrieve_result.invoke({
#       "file_list": ",".join(file_path_list),
#       "question": "图像局部重绘",
# }))

# print(format_retrieve_result.invoke({
#       "file_list": ",".join(file_path_list),
#       "question": "python中如何调用图像局部重绘",
# }))
# print(type(format_retrieve_result.invoke({
#        "file_list": ",".join(file_path_list),
#        "question": "python中如何调用图像局部重绘",
#  })))

"""
question
chat_history
file_list
"""
prompt_chain = (RunnablePassthrough.assign(retrieve_info=format_retrieve_result)
      |  RunnablePassthrough.assign(
        # 如果需要检索（retrieve_flag为Y），则调用inject_retrieve_content生成包含检索内容的提示词
        # 否则直接使用原始问题作为user_content
        user_content=lambda x: (
            inject_retrieve_content(
                x["question"], x["retrieve_info"]["code_flag"] == "Y"
            )
            if x["retrieve_info"]["retrieve_flag"] == "Y"
            else x["question"]
        )
    ) |  ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""你是一个智能助手，你需要帮助用户解答问题"""
                ),  # 系统消息，定义助手角色
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # 对话历史占位符，将插入对话历史
                HumanMessagePromptTemplate.from_template(
                    "{user_content}"
                ),  # 用户内容占位符，将插入处理后的用户提示词
            ]
        ))

# print(prompt_chain.invoke({
#   "question": "python中如何调用图像局部重绘",
#   "chat_history": [],
#   "file_list": ",".join(file_path_list)
# }))
def test_log(x):
  # print('x', x)
  # messages = x.messages
  # for message in messages:
  #   content = message.content
  #   msg_type = message.type
  #   print(content, msg_type)
  return x
# create_llm("local")
chain = (RunnablePassthrough.assign(
  chat_history= lambda  x: get_history_list(),
  file_list = lambda x: ",".join(file_path_list)
) | prompt_chain | test_log | create_llm("local") | StrOutputParser())
#

# print(chain.invoke({
#     "question": "python中如何调用图像局部重绘"
# }))

# print(chain.invoke({
#     "question": "你是谁"
# }))
while True:
    question = input("\n请输入你的问题: ")
    if question.lower() == "exit":
        break

    # 流式生成回答并实时打印
    full_text = ""  # 用于存储完整的回答文本
    # 调用链的stream方法，以流式获取输出，传入用户问题
    for chunk in chain.stream(
        {
            "question": question,
        }
    ):
        full_text += chunk  # 累加每个输出块到完整文本
        print(chunk, end="", flush=True)  # 实时打印每个输出块，不换行，立即刷新

    # 将用户问题添加到对话历史
    save_message(HumanMessage(content=question))
    # 将完整回答添加到对话历史
    save_message(AIMessage(content=full_text))

    print_history_list()
