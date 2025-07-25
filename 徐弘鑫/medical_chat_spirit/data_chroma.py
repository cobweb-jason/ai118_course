from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PDFMinerLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import find_dotenv, load_dotenv
from langchain_core.runnables import RunnableLambda

import numpy as np
import pymysql
import dashscope
import json
import os

class MYmysql():
    load_dotenv(find_dotenv())
    dashscope_api_key = os.getenv("aliyun_api_key")
    def __init__(self):
        # 设置 OpenAI API 密钥
        load_dotenv(find_dotenv())
        dashscope_api_key = os.getenv("aliyun_api_key")


    def create_mysql_connection(self):
        return pymysql.connect(
            host='localhost',
            user='jason',
            password='46528138',
            database='medical_spirit',
        )

    # 使用 OpenAI API 把查询文本（用户输入）转换为向量。
    def get_query_embedding(self, query):
        response = dashscope.TextEmbedding.call(
            model='text-embedding-v1',
            input=[query],
            api_key=self.dashscope_api_key
        )
        if response.status_code == 200:
            return np.array([response['output']['embeddings'][0]['embedding']])
        else:
            raise Exception(f"向量化失败，错误信息: {response}")# 转换为二维数组列表

    # 从 MySQL 取出存储的向量，计算与查询向量的余弦相似度，按相似度排序返回结果。
    def retrieve_similar_items(self, query, top_n=3):
        conn = self.create_mysql_connection()
        cursor = conn.cursor()
        query_embedding = self.get_query_embedding(query)

        # 获取数据库中所有表名
        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        all_results = []
        for table in tables:
            table_name = table[0]
            print(f"查询表: {table_name}")
            try:
                # 尝试从每个表中查询 content 和 embedding 字段
                select_query = f"SELECT content, embedding FROM {table_name}"
                cursor.execute(select_query)
                results = cursor.fetchall()
                all_results.extend(results)
            except pymysql.Error as e:
                print(f"查询表 {table_name} 时出错: {e}")

        print("查询结果:", all_results)
        similarities = []
        for text, embedding_str in all_results:
            embedding = np.array(eval(embedding_str))
            similarity = cosine_similarity(query_embedding, [embedding])[0][0]
            similarities.append((text, similarity))


        similarities.sort(key=lambda x: x[1], reverse=True)
        cursor.close()
        conn.close()
        return similarities[:top_n]

    # 实现文件夹下的所有文件进行读取、分片和embedding向量化并存储至MySQL数据库中以文件名为表名的表中
    # 存储格式为id, content, embedding
    def splitter_save (self,file_name,embeddings):
        db = pymysql.connect(host='localhost',user='jason',password='46528138',port=3306) #连接数据库
        cursor = db.cursor()  #获取操作游标
        cursor.execute('use medical_spirit;')
        # 修改表结构，添加 embedding 列
        table_name = os.path.splitext(os.path.basename(file_name))[0]
        cursor.execute(f'CREATE TABLE if NOT EXISTS {table_name} (id VARCHAR(255) NOT NULL, content TEXT NOT NULL, embedding TEXT NOT NULL, PRIMARY KEY(id));')


        if os.path.splitext(file_name)[1] == ".docx":
            document = Docx2txtLoader(file_name).load()
            splits = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 40).split_documents(document)
            print(splits)
            for i, split in enumerate(splits):
                content = split.page_content
                embedding = embeddings.embed_query(content)
                # 将 embedding 转换为 JSON 字符串
                embedding_str = json.dumps(embedding)
                # 插入数据到数据库
                cursor.execute(f"INSERT INTO {table_name} (id, content, embedding) VALUES (%s, %s, %s)", (str(i), content, embedding_str))
            db.commit()

        elif os.path.splitext(file_name)[1] == ".pdf":
            document = PDFMinerLoader(file_name).load()
            splits = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 40).split_documents(document)
            print(splits)
            for i, split in enumerate(splits):
                content = split.page_content
                embedding = embeddings.embed_query(content)
                # 将 embedding 转换为 JSON 字符串
                embedding_str = json.dumps(embedding)
                # 插入数据到数据库
                cursor.execute(f"INSERT INTO {table_name} (id, content, embedding) VALUES (%s, %s, %s)", (str(i), content, embedding_str))
            db.commit()
        db.close()


    def file_data_embedding(self,folder_path):
        embeddings = DashScopeEmbeddings(
            dashscope_api_key = self.dashscope_api_key,
            model = "text-embedding-v1",
        )
        # chroma_db = Chroma(embedding_function=embedding)
        for root,dirs,filenames in os.walk(folder_path):
            for filename in filenames:
                file_path = os.path.join(root,filename)
                self.splitter_save(file_path,embeddings)

    def get_mysql_retriever(self):
        return RunnableLambda(lambda query: [item[0] for item in self.retrieve_similar_items(query)])

    
    
if __name__ == '__main__':
    mysql_instance = MYmysql()
    informations = mysql_instance.file_data_embedding("F:\CODES\medicine _spirit\medical_spirit_latest\chat_spirit\data") # 以指定文件为内容，创建一个检索器实例
    retriever = mysql_instance.get_mysql_retriever()
    query = "糖尿病怎么治疗？"
    results = retriever.invoke(query)
    print("检索结果:", results)
    # MYmysql().file_data_embedding('F:\CODES\medicine _spirit\medical_spirit_latest\chat_spirit\data')
    