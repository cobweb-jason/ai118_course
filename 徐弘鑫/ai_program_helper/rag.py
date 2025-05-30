# 从 zhipuai 模块导入 ZhipuAI 类，用于调用智谱 AI 的 API
from zhipuai import ZhipuAI
# 从 transformers 库导入 AutoModel 和 AutoTokenizer 类，用于加载预训练模型和分词器
from transformers import AutoModel, AutoTokenizer
# 从 embedding 模块导入 embed_with_qwen 和 embed_with_bge 函数，用于生成文本的向量表示
from embedding import embed_with_qwen, embed_with_bge
# 导入 chromadb 库，用于与向量数据库进行交互
import chromadb

# 定义一个生成器函数，用于流式调用 ChatGLM 模型
def chatglm_stream(input_text:str,history_list:list):
    # 初始化 ZhipuAI 客户端，需要提供 API 密钥
    client = ZhipuAI(api_key='baf3e0451a944f4d8d1b0fad9f89edf8.1JMuZDQspIp5n4QC')
    # 初始化消息列表，将历史对话添加到消息列表中
    messages = history_list
    # 将用户输入的文本添加到消息列表中
    messages.append({'role':'user','content':input_text})
    # 调用智谱 AI 的 chat.completions.create 方法，以流式方式获取模型的回复
    response = client.chat.completions.create(
        model='glm-4',
        messages=messages,
        stream=True
    )
    # 遍历流式响应，逐块生成回复内容
    for chunk in response:
        content = chunk.choices[0].delta.content
        yield content
    # 将用户输入的文本添加到历史对话列表中
    history_list.append({'role': 'assistant', 'content': input_text})
    # 再次调用智谱 AI 的 chat.completions.create 方法，以流式方式获取模型的回复
    response = client.chat.completions.create(
        model='glm-4',
        messages = messages,
        stream=True
    )
    # 遍历流式响应，逐块生成回复内容
    for chunk in response:
        content = chunk.choices[0].delta.content
        yield content

# 定义一个机器人类，用于实现 RAG（检索增强生成）功能
class Robot():
    def __init__(self):
        # 初始化预训练模型（当前代码注释掉，未实际使用）
        # self.model = AutoModel.from_pretrained('')
        # 初始化分词器（当前代码注释掉，未实际使用）
        # self.tokenizer = AutoTokenizer.from_pretrained('')
        # 创建一个与本地向量数据库的 HTTP 客户端连接
        self.client = chromadb.HttpClient(host='localhost', port=8000)

    # 定义 RAG 方法，用于从向量数据库中检索相关信息
    def RAG(self,input_text,dbname):
        # 使用 embed_with_qwen 函数生成用户输入文本的向量表示
        search_embedding = embed_with_qwen([input_text])
        # 从向量数据库中获取指定名称的集合
        collection = self.client.get_collection(name=dbname)
        # 从向量数据库中召回 3 个最相关的文本
        result = collection.query(
        query_embeddings=search_embedding,
        n_results=3
        )
        # 初始化一个空字符串，用于存储检索结果
        result_all = ''

        # 遍历检索结果，将文档和对应的答案拼接成字符串
        for i in range(len(result['documents'][0])):
            result_docments = result['documents'][0][i]
            result_answer = result['metadatas'][0][i]['answer']
            result_all += f'{result_docments}\n [参考资料] : {result_answer}\n\n'
        return result_all

    # 定义测试运行方法，用于与用户进行交互测试
    def run_test(self):
        # 初始化历史对话列表，设置系统角色的提示信息
        history_list = [{'role':'system','content':'你是一个智能助教,对于用户提出的问题，你需要根据给出的【参考资料】对问题进行回答。你的回答需要按照以下两个步骤：1.分析用户问题和参考资料，判断是否有【参考资料】可以解答用户的问题，如果有则说明【参考资料】的名称，如果没有，则首先告知用户没有任何可参考的资料，需要注意答案的准确性。2.根据资料内容对提问进行解答。'}]

        while True:
            # 获取用户输入
            input_text = input('请输入:')
            # 如果用户输入 exit，则退出循环
            if input_text == 'exit':
                break

            # 指定向量数据库集合的名称
            db_name = 'collection01'
            # 调用 RAG 方法，从向量数据库中检索相关信息
            result_rag = self.RAG(input_text, db_name)
            # 打印检索结果
            print(result_rag)
            # 将用户输入和检索结果拼接成新的输入文本
            input_text = input_text + f'参考资料: \n{result_rag}'
            # 调用 chatglm_stream 函数，获取模型的流式回复
            result = chatglm_stream(input_text, history_list)
            # 初始化一个空字符串，用于存储模型的完整回复
            result_all = ''
            # 遍历流式回复，逐块打印并拼接成完整回复
            for res in result:
                result_all += res
                print(res, end='', flush=True)
            print('')
            # 将用户输入添加到历史对话列表中
            message_input = {'role': 'user', 'content': input_text}
            # 将模型的完整回复添加到历史对话列表中
            message_output = {'role': 'assistant', 'content': result_all}
            history_list.append(message_input)
            history_list.append(message_output)

if __name__ == '__main__':
    # 创建 Robot 类的实例
    robot = Robot()
    # 调用 run_test 方法，开始与用户进行交互测试
    robot.run_test()