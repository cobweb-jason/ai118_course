import streamlit as st
from my_robot import Robot
import funcs
from data_chroma import MYmysql
from dotenv import load_dotenv
from uuid import uuid4


def init_interface():
    st.set_page_config(layout="wide")   # 设置页面布局为宽屏
    mysql_instance = MYmysql()

    # # 保存相关公共对象
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())  # 初始化会话ID为一个新的UUID
    informations = mysql_instance.file_data_embedding("F:\CODES\medicine _spirit\medical_spirit_latest\chat_spirit\data") # 以指定文件为内容，创建一个检索器实例
    retriever = mysql_instance.get_mysql_retriever()    # 创建检索器实例
    st.session_state.robot = Robot(retriever=retriever)    # 初始化Robot类，传入session_id和检索器实例

    st.title("Molly医疗精灵")   # 设置标题

    with st.sidebar:    # 设置侧边栏
        st.header(f"当前对话ID：{st.session_state.session_id}") # 显示当前对话ID

        st.button("新建对话", on_click=funcs.start_session) # 创建新对话按钮，点击后调用start_session函数

        all_session = funcs.get_all_session_ids()   # 获取所有会话ID

        for sid in all_session: # 遍历所有会话ID
            with st.expander(f"对话ID：{sid}"): # 创建可展开的对话ID面板，用于显示会话内容
                col1, col2 = st.columns(2)  # 创建两列布局
                col1.button("继续对话", key=f"continue_{sid}", on_click=funcs.continue_session,args=(sid,)) # args=(sid,)表示传入continue_session函数的参数为当前会话ID
                col2.button("删除对话", key=f"delete_{sid}", on_click=funcs.delete_session,args=(sid,))
                
                message = funcs.get_session_message(sid)   # 获取当前会话的消息内容
                with st.container():    # 显示当前会话的消息内容
                    for msg in message: # 遍历消息内容
                        with st.chat_message(msg[0]):   # 根据角色显示消息
                            st.write(msg[1])

    messages = funcs.get_session_message(st.session_state.session_id)  # 获取当前会话的消息内容
    with st.container():    # 显示当前会话的消息内容
        for msg in messages:    # 遍历消息内容
            with st.chat_message(msg[0]):   # 根据角色显示消息
                st.write(msg[1])    # 显示消息内容

    question = st.chat_input("Say something")   # 用户输入问题的聊天输入框

    # 根据输入的内容判断是否进行对话
    if question is not None:    # 判断用户是否输入了问题
        resp = funcs.create_response(question,st.session_state.session_id)  # 调用create_response函数，传入用户输入的问题和会话ID，获取AI的回复

        with st.chat_message("human"):  # 显示用户输入的问题
            st.write(question)    # 显示用户输入的问题内容

        st.chat_message("ai").write_stream(resp)  # 显示AI的回复内容，使用流式输出方式


if __name__ == "__main__":
    load_dotenv()
    init_interface()
