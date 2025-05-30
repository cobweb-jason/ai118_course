# 导入 dashscope 库，用于调用 DashScope 提供的 API
import dashscope
# 导入 PyTorch 库，用于深度学习相关操作
import torch

# 定义一个函数，使用 Qwen 模型为输入的文本列表生成向量表示
def embed_with_qwen(text_list):
    # 设置 DashScope 的 API 密钥，用于身份验证
    dashscope.api_key = 'sk-3885cf7d76b54a74b2f5370a8a3c4529'
    # 初始化一个空列表，用于存储生成的向量表示
    embedding_list = []
    # 遍历输入的文本列表
    for text in text_list:
        # 调用 DashScope 的文本嵌入 API，使用 text_embedding_v2 模型
        resp = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v2,
            input=text,
        )
        # 检查 API 调用是否成功
        if resp.status_code == 200:
            # 从响应中提取向量表示并添加到列表中
            embedding = resp.output['embeddings'][0]['embedding']
            embedding_list.append(embedding)
        else:
            # 若 API 调用失败，打印响应信息
            print(resp)
    # 返回生成的向量表示列表
    return embedding_list

# 定义一个函数，使用 BGE 模型为输入的文本列表生成向量表示
def embed_with_bge(text_list, tokenizer, model):
    # 使用分词器对输入的文本列表进行编码，设置最大长度为 512，进行填充和截断操作
    encoded_input = tokenizer(text_list, max_length=512, padding=True, truncation=True, return_tensors='pt')

    # 禁用梯度计算，减少推理阶段的内存消耗
    with torch.no_grad():
        # 将编码后的输入传入模型，得到模型输出
        model_output = model(**encoded_input)

    # 提取句子的嵌入向量，取输出的第一个维度的第一个元素
    sentence_embeddings = model_output[0][:, 0]
    # 将嵌入向量转换为列表并返回
    return sentence_embeddings.tolist()

if __name__ == '__main__':
    # 调用 embed_with_qwen 函数，对示例文本进行向量表示生成并打印结果
    print(embed_with_qwen(['我们一起学习RAG课程']))