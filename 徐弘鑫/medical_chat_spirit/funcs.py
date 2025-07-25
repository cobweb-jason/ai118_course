import streamlit as st
from langchain_core.messages import HumanMessage
from uuid import uuid4

# 用于获取和管理聊天会话的函数
def get_session_message(session_id:str=None):
    messages = []# 创建当前对话消息内容列表
    default_session_id = st.session_state.get("session_id", "chat_01")  # 设置默认会话id，如果session_id为None，则使用当前会话id“chat_01”
    history = st.session_state.robot.get_history(session_id or default_session_id)  # 获取当前会话的历史消息
    for message in history.messages[1:]:# 将消息历史转换为元组列表，其中元组第一项为消息发送者
        if isinstance(message, HumanMessage):
            message = ("Human", message.content)
        else:
            message = ("AI", message.content)
        # message = ("Human", message.content) if isinstance(message, HumanMessage) else ("AI", message.content)
        messages.append(message)
    return messages

# 用于创建回复的函数
def create_response(question, session_id):
    # session_id = st.session_state.session_id
    return st.session_state.robot.stream(question, session_id)  # 调用robot函数中的stream方法生成回复

# 用于开始新会话的函数
def start_session():
    # 创建一个新的会话ID
    st.session_state.session_id = str(uuid4())  # 新建一个UUID作为会话ID并存入到session_state中
    st.session_state.robot.get_history(st.session_state.session_id) # 获取当前会话的历史消息（会话ID为新创建的UUID，内容应为所设定提示词）

# 用于获取所有会话ID的函数
def get_all_session_ids():
    return st.session_state.robot.check_session_id()  # 调用robot函数中的check_session_id方法获取所有会话ID

# 用于继续会话的函数
def continue_session(session_id):
    st.session_state.session_id = session_id

# 用于删除会话的函数
def delete_session(session_id):
    st.session_state.robot.get_history(session_id).clear()  # 清楚当前会话的历史消息

    if session_id == st.session_state.session_id:   # 如果删除的是当前会话，则将session_id设置为None
        st.session_state.session_id = None
