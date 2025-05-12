from openai import OpenAI
import gradio as gr
def generate_response(prompt, history=None):
    """
    生成基于 DeepSeekAPI 的聊天回复
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
 
    response = OpenAI.ChatCompletion.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
    )
 
    reply = response.choices[0].message.content
    if not history:
        history = []
    history.append({"role": "user", "content": prompt})
    history.append({"role": "assistant", "content": reply})
    return reply, history
def chat_with_gradio(user_input, conversation_state=gr.State([]), api_key="sk-dd0ef6a1cfb248e4b261c61358aef96c", base_url="https://api.deepseek.com", model_id="deepseek-chat"):
    """
    Gradio 界面的交互函数
    """
    # 获取当前会话状态
    conversation_history = conversation_state
    # 调用聊天函数
    reply = generate_response(user_input,conversation_history)
    # 更新会话历史
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": reply})
    # 返回机器人回复和更新后的会话历史
    return reply
# 创建 Gradio 界面
iface = gr.Interface(
    fn=chat_with_gradio,
    inputs=[
        gr.Textbox(lines=2, placeholder="请输入...", label="你的问题"),
        gr.State([])  # 用于存储会话历史
    ],
    outputs=[
        gr.Textbox(label="机器人回答"),
        gr.State()  # 会话历史（用户不可见）
    ],
    title="聊天机器人",
    description="基于 DeepSeekAPI 的聊天机器人",
    examples=[
        ["给我讲一个科幻故事吧！"],
        ["如何提高编程能力？"]
    ]
)
 
# 启动界面
iface.launch(share=True)