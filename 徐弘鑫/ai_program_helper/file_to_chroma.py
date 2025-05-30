# 导入 chromadb 库，用于与向量数据库进行交互
import chromadb
# 从 embedding 模块导入 embed_with_qwen 函数，用于生成文本的向量表示
from embedding import embed_with_qwen
# 导入 json 模块，用于处理 JSON 格式的数据
import json

# 定义一个函数，用于解析 JSON 文件
def json_parse(file):
    # 以只读模式打开指定的 JSON 文件，并使用 utf-8 编码
    with open(file, 'r', encoding='utf-8') as f:
        # 加载 JSON 文件内容到 Python 数据结构中
        data = json.load(f)

        # 初始化一个空列表，用于存储解析后的结果
        result = []
        # 遍历 JSON 文件中的每个元素
        for item in data:
            # 从每个元素的 'k_qa_content' 字段中提取关键词和答案，以 '#' 为分隔符
            keyword, answer = item['k_qa_content'].split('#', maxsplit=1)
            # 将关键词和答案作为一个列表添加到结果列表中
            result.append([keyword, answer])  # 用元组或列表
        return result

# 定义一个主函数，用于将文件内容处理后存储到向量数据库中
def run(file_path,dbname):
    # 调用 json_parse 函数解析指定的文件
    text_list = json_parse(file_path)
    # 初始化一个空列表，用于存储关键词
    keyword_list = []
    # 初始化一个空列表，用于存储答案
    answers_list = []
    # 遍历解析后的文本列表
    for text in text_list:
        # 将每个文本中的关键词添加到关键词列表中
        keyword_list.append(text[0])
        # 将每个文本中的答案以字典形式添加到答案列表中
        answers_list.append({'answer':text[1]})
    # 使用闭源 API 调用，生成关键词列表的向量表示
    embedding_list = embed_with_qwen(keyword_list)
    # 使用开源模型调用（此处为注释掉的示例代码）
    # model = AutoModel.from_pretrained('')
    # tokenizer = AutoTokenizer.from_pretrained('')
    # embedding_list = embed_with_bge(keyword_list,tokenizer,model)
    # 检查生成的向量表示是否为列表类型
    if type(embedding_list)==list:
        # 创建一个与本地向量数据库的 HTTP 客户端连接
        client = chromadb.HttpClient(host='localhost', port=8000)
        # 获取或创建指定名称的向量数据库集合
        collection = client.get_or_create_collection(name=dbname)
        # 初始化一个空列表，用于存储每个向量的唯一标识
        ids_list = []
        # 遍历向量列表，为每个向量生成一个唯一标识
        for i in range(len(embedding_list)):
            ids_list.append(str(dbname)+str(i))
        # 将向量、元数据、文档等信息添加到向量数据库集合中
        collection.add(
        ids=ids_list,
        embeddings=embedding_list,
        metadatas=answers_list,
        documents=keyword_list
    )
    else:
        return 'error'

    # 打印向量化完成的提示信息
    print('向量化完成')
    # print(collection.get(include=['documents', 'metadatas', 'embeddings']))
    return 'success'

if __name__ == '__main__':
    # 创建一个与本地向量数据库的 HTTP 客户端连接
    client = chromadb.HttpClient(host='localhost', port=8000)
    # 获取向量数据库中的所有集合
    collections = client.list_collections()
    # 遍历所有集合，删除每个集合
    for collection in collections:
        client.delete_collection(collection.name)
    # 指定要处理的 JSON 文件路径
    file_path = 'F:/CODES/RAG/chat_box/test.json'
    # 指定要创建的向量数据库集合名称
    dbname = 'collection01'
    # 调用 run 函数并打印结果
    print(run(file_path, dbname))