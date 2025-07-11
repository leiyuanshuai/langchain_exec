from typing import List, Union
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
def create_history_manage():
    history_list: List[Union[SystemMessage, HumanMessage, AIMessage]] = []

    def save_message(message: Union[SystemMessage, HumanMessage, AIMessage]):
        if isinstance(message, (SystemMessage, HumanMessage, AIMessage)):
            history_list.append(message)
        else:
            raise TypeError("message must be SystemMessage or HumanMessage or AIMessage")

    def get_history_list():
        return history_list
    def print_history_list():
        print("\n消息对话开始历史开始", "-" * 20)
        for message in history_list:
          print(f"{message.type}: {message.content}")
        print("\n消息对话历史结束", "-" * 20)
    return save_message,get_history_list, print_history_list
