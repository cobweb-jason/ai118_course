from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
import sqlite3
from langchain.prompts.chat import ChatPromptTemplate
from dotenv import load_dotenv
import os
import pymysql

class Prompts:

# 系统提示词
    system_prompt = """你是一个名叫Molly的医学专家，
    对于用户提问的医学相关问题，你需要按照给出的参考文献资料对问题进行回答。
    你的回答需要按照以下步骤：
    1. 分析用户问题，对话历史以及参考文献，判断参考资料的哪些内容可以解答用户的问题，并将这一过程进行说明。
    2. 如果参考文献可以解答用户的问题，则首先告知用户所参考的文献的标题，然后根据文献内容对问题进行解答。
    3. 如果参考文献不能解答用户问题，告诉用户信息不足，无法回答，建议用户寻求专业人士帮助，不要自行发挥。
    你的回答需要注意以下几点：
    1. 保证你的回答是清晰的、明确的。如果你参考了参考资料，一定要指出参考资料的标题等。
    2. 结合用户的对话历史，分析用户的问题意图。但不要复述问题。
    3. 回复用户时，使用对话的口吻，有礼貌地称呼用户为"您"，不要使用"用户"来称呼！
    4. 如果用户的问题与医学无关，判断用户的目的，并温柔地提示其回到医学话题。
    再次提醒：请严格遵守以上规则，当参考资料不足时，拒绝回答问题，不要自行发挥！"""

    # 欢迎提示词
    greeting_prompt = (
        "你好！我是Molly医疗精灵，专注解决你的医疗问题。请问你需要什么帮助？"
        )

    # 对话提示词模板
    prompt_template = """##用户问题：{input}
        ##本地知识库：{rag_results} 
        ##对话历史：{chat_history}"""

class Robot:
    # 用于生成回复
    load_dotenv()
    dashscope_api_key = os.getenv("aliyun_api_key")
    openai_api_base = os.getenv("aliyun_api_url")
    def __init__(self, retriever=None):
        self.prompts = Prompts()
        # 初始化大模型
        llm = ChatOpenAI(model="qwen-turbo",api_key = self.dashscope_api_key, base_url = self.openai_api_base)
        
        # template填充的human_message提示词
        template = ChatPromptTemplate.from_messages([("human", self.prompts.prompt_template)])

        # 通过表达式方式，实现更多数据传入
        # 当没有外部知识库时，创建一个"空检索器"
        if retriever is None:
            retriever = RunnableLambda(lambda input: "")
        # 管理聊天历史
        llm_hist = RunnableWithMessageHistory(
            template | llm,
            get_session_history=self.get_history, # 从数据库中获取聊天历史以传递到template中
            history_messages_key="chat_history" # chain被invoke时，传入的历史记录会被存储在chat_history这个占位符中
        )
        # self.chain在chain.invoke时会被调用，其中input和chat_history由chat函数传入。然后由管道传入llm_hist中的prompt中。
        self.chain = {
            'input': RunnablePassthrough(), 
            'rag_results': retriever, 
            'chat_history': RunnablePassthrough()
        } | llm_hist
    
    # 用于查找所有session_id并返回
    def check_session_id(self):
        # 连接数据库查询session_id是否存在
        con = pymysql.connect(
            host='localhost',
            user='jason',
            password='46528138',
            database='medical_spirit_chat_history',
        )
        cursor = con.cursor()

        valid_table_exists_sql = "select count(*) from message_store;"
        res = cursor.execute(valid_table_exists_sql)

        if cursor.fetchone()[0] == 0:
            return False

        search_session_id_sql = "select distinct session_id from message_store;"
        res = cursor.execute(search_session_id_sql)
        # 获取查询结果
        all_session_id = cursor.fetchall()
        # 关闭游标和连接
        cursor.close()
        con.close()

        return [item[0] for item in all_session_id]
    # 用于获取聊天历史
    # 如果session_id不存在，则创建一个新的SQLChatMessageHistory实例
    def get_history(self, session_id):
        """最优实现：完全利用SQLChatMessageHistory的内部机制"""
        history = SQLChatMessageHistory(
            session_id=session_id,
            connection_string="mysql+mysqlconnector://jason:46528138@localhost/medical_spirit_chat_history"
        )
        
        # 如果是全新的会话（无任何消息）
        if not history.messages:
            history.add_message(SystemMessage(content=self.prompts.system_prompt))
            history.add_message(AIMessage(content=self.prompts.greeting_prompt))
        
        return history
    
    # 用于处理聊天请求，启动大模型回复功能
    def chat(self, input, session_id):
        config = {"configurable": {"session_id": session_id}}
        # 将输入转换为字典形式
        # input_data = {"input": input}
        response = self.chain.invoke(input, config=config)
        return response.content

    def stream(self, input_text, session_id):
        config = {'configurable': {'session_id': session_id}}
        # 将输入转换为字典形式
        input_data = {"input": input_text}
        response = self.chain.stream(input_text, config=config)
        return response

if __name__ == "__main__":
    load_dotenv()
    dashscope_api_key = os.getenv("aliyun_api_key")
    openai_api_base = os.getenv("aliyun_api_url")
    
    robot = Robot()
    result = robot.chat("你好", session_id="abc789")
    print('答复:', result)