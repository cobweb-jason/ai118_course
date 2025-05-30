import gradio as gr  # 导入Gradio库，用于快速搭建交互式Web界面
from rag import Robot  # 导入自定义的Robot类，用于实现RAG（检索增强生成）功能
from zhipuai import ZhipuAI  # 导入智谱AI SDK，用于调用大语言模型
from file_to_chroma import run  # 导入文件入库函数，将文件内容向量化并存入Chroma数据库
from chromadb import HttpClient  # 导入Chroma数据库的HTTP客户端，用于与Chroma服务进行交互


# 处理用户输入，将用户问题添加到聊天历史中，并在历史为空时添加系统提示
# :param question: 用户输入的问题，字符串类型
# :param history: 聊天历史记录，列表类型，每个元素为包含角色和内容的字典
# :return: 清空后的输入框内容（空字符串）和更新后的聊天历史
def user(question: str, history: list):
    # 如果历史记录为空，添加系统提示信息
    if len(history) == 0:
        history.append({'role': 'system',
                        'content': '你是一个智能助教,对于用户提出的问题，你需要根据给出的【参考资料】对问题进行回答。你的回答需要按照以下两个步骤：1.分析用户问题和参考资料，判断是否有【参考资料】可以解答用户的问题，如果有则说明【参考资料】的名称，如果没有，则首先告知用户没有任何可参考的资料，需要注意答案的准确性。2.根据资料内容对提问进行解答。'})
    # 将用户输入添加到聊天历史中
    history.append({'role': 'user', 'content': question})
    return "", history  # 返回空输入框和更新后的历史


# 机器人回复函数，负责执行RAG检索并调用大语言模型生成回复
# :param history: 聊天历史记录，列表类型，每个元素为包含角色和内容的字典
# :param dbname: 要使用的数据库名称，字符串类型
# :return: 逐步生成的聊天历史，使用生成器返回以实现流式响应
def bot(history: list, dbname):
    input_text = history[-1]['content']  # 获取用户最新输入的问题
    print('\n\n用户输入：', input_text)  # 打印用户输入到控制台，便于调试
    result_rag = robot.RAG(input_text, dbname)  # 调用RAG机器人进行知识库检索，获取参考资料
    # 将参考资料拼接到用户输入后面，作为大模型的输入
    print('\n\n大模型输入：', result_rag)
    history[-1]['content'] = input_text + f'参考资料: \n{result_rag}'
    client = ZhipuAI(api_key='baf3e0451a944f4d8d1b0fad9f89edf8.1JMuZDQspIp5n4QC')  # 初始化智谱AI客户端，使用指定的API密钥
    messages = history  # 将聊天历史作为与大模型交互的消息列表
    # 调用GLM-4大模型，启用流式生成回复
    response = client.chat.completions.create(
        model='glm-4',
        messages=messages,
        stream=True
    )
    print('\n\n大模型输出：', response)  # 打印大模型的响应，便于调试
    # 下面两行会把“参考资料”从历史中移除，只保留原始问题（如需保留可注释掉）
    parts = history[-1]['content'].split('参考资料', 1)
    print('\n\nparts:', parts[0])
    history[-1]['content'] = parts[0].strip()
    print('\n\nhistory[-1][content]:', history[-1]['content'])
    # 添加助手回复占位符到聊天历史中
    history.append({'role': 'assistant', 'content': ''})
    print('\n\nhistory:', history)
    # 实时追加大模型输出到聊天历史，并通过生成器返回给Gradio前端
    for chunk in response:
        print('\n\nchunk:', chunk)  # 打印每个响应块，便于调试
        print('\n\nresponse:', response)  # 打印完整响应，便于调试
        history[-1]['content'] += chunk.choices[0].delta.content
        print('\n\n大模型输出内容：', chunk.choices[0].delta.content)
        yield history


# 获取Chroma数据库中的所有集合名称，作为可用的数据库列表
# :return: 数据库集合名称列表，如果出错则返回空列表
def get_db_list():
    try:
        client = HttpClient(host='localhost', port=8000)  # 连接本地运行的Chroma服务
        collections = client.list_collections()  # 获取Chroma数据库中的所有集合
        return [collection.name for collection in collections]  # 返回集合名称列表
    except Exception as e:
        print(f'获取数据库列表时出错: {e}')  # 打印错误信息
        return []  # 出错时返回空列表


# 构建数据库，将上传的文件内容向量化并存储到Chroma数据库中
# :param datafile: 上传的文件路径，字符串类型
# :param dbname: 要创建的数据库名称，字符串类型
# :return: Gradio的信息提示组件，显示数据库创建成功或失败的消息
def build_db(datafile, dbname):
    try:
        res = run(datafile, dbname)  # 调用文件入库函数，将文件内容存入指定数据库
        if res == 'success':
            return gr.Info('数据库创建成功', duration=5)  # 数据库创建成功，显示提示信息
        else:
            return gr.Error('数据库创建失败', duration=5)  # 数据库创建失败，显示错误信息
    except Exception as e:
        print(f'数据库创建时出错: {e}')  # 打印错误信息
        return gr.Error('数据库创建失败:', duration=5)  # 出现异常时显示错误信息


# 刷新数据库下拉选择框的选项，重新获取最新的数据库列表
# :return: 更新后的Gradio下拉选择框组件，包含最新的数据库列表
def refresh_db_choices():
    dbList = get_db_list()  # 获取最新的数据库列表
    return gr.update(choices=dbList, multiselect=False)  # 更新下拉选择框的选项


# 初始化数据库列表，获取当前可用的数据库集合名称
dbList = get_db_list()
# 实例化RAG机器人，用于后续的检索和生成任务
robot = Robot()

# 使用Gradio的Blocks界面构建器创建交互式界面
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown('# 智能学习助教')  # 显示界面标题
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label='对话框', type='messages')  # 创建聊天对话框组件，用于显示聊天历史
            question = gr.Textbox(label='请输入')  # 创建文本输入框，用于用户输入问题
            clear_button = gr.Button("clear")  # 创建清空按钮，用于清空聊天历史
            examples = gr.Examples(['介绍python', '卓越领导力的五种行为'], inputs=[question])  # 创建示例问题组件，用户点击示例可快速填充输入框
        with gr.Column():
            gr.Markdown('### 数据库构建')  # 显示数据库构建模块标题
            datafile = gr.File(type='filepath', label='上传文件')  # 创建文件上传组件，用户可上传文件
            dbname = gr.Textbox(label='数据库名称')  # 创建文本输入框，用于输入要创建的数据库名称
            build_button = gr.Button('开始构建')  # 创建构建按钮，点击后触发数据库构建操作
            gr.Markdown('### 数据库选择')  # 显示数据库选择模块标题
            dbchoose = gr.Dropdown(choices=dbList, label='数据库名称')  # 创建下拉选择框，用于选择要使用的数据库
            dbrefresh = gr.Button('刷新')  # 创建刷新按钮，点击后刷新下拉选择框的选项

        # 绑定事件：用户在输入框提交问题时，先调用user函数处理输入，再调用bot函数生成回复
        question.submit(user, [question, chatbot], [question, chatbot], queue=False).then(
            bot, [chatbot, dbchoose], chatbot, queue=False
        )
        # 绑定事件：点击清空按钮时，清空聊天对话框的内容
        clear_button.click(lambda: None, None, chatbot, queue=False)
        # 绑定事件：点击构建按钮时，调用build_db函数进行数据库构建
        build_button.click(build_db, inputs=[datafile, dbname], outputs=[])
        # 绑定事件：点击刷新按钮时，调用refresh_db_choices函数刷新下拉选择框的选项
        dbrefresh.click(refresh_db_choices, outputs=dbchoose)

demo.launch()  # 启动Gradio应用，将界面部署到本地服务器