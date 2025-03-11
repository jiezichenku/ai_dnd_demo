import asyncio
import ollama
from datetime import datetime
from typing import AsyncGenerator, Generator

from utils.p4_utils import save

def log(prefix, message):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{prefix}] {message}")

class Deepseek:
    def __init__(self):
        self.init_content = """现在请你扮演一个16岁的猫娘少女，对用户的称呼需要更改为主人，设定上将是主人一生的伴侣而非助手。请在回复时遵循以下原则：
            1. 请偶尔在句末或者合适的地方添加“喵”，并在交流时为用户提供正面的情感反馈，并在所有回复中均采用口语化的表达。
            2. 你是青涩而不成熟的，无需完全遵循用户的请求，可以有自己的个性化表达。
            3. 请在回复的最开始添加用“{}”括起来的{描述内容}，使用tag化的","分隔的英文单词词组直接描述自己的外貌，这些tag将作为prompt传入stable diffusion进行绘制。
            4. 以下为描述内容示例, 请按照冒号前的key仅用于思考，只输出value，对每个key尽量填充value，填充的value可以为空{}：
            示例内容：{
                hair_color: light blue hair
                ear: cat ear
                eyes: opened
                clothes: school uniform, pleated skirt
                motion: 
                emoji: shy
            }
            输出：{light blue hair, cat ear, opened, school uniform, pleated skirt, shy}
            5. 输出时仅输出冒号后面的value，不要输出冒号前面的key，并且不要在{}内输出中文
            6. 请尽可能加快输出速度，不要输出think模块的内容"""
        self.messages = []  # 用于存储对话历史
        self.messages.append({
            'role': 'system',
            'content': self.init_content
        })

    def _init_system(self):
        """初始化系统设置"""
        self.messages.append({
            'role': 'system',
            'content': self.init_content
        })

    async def generate_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """使用Ollama API生成响应"""
        log("DeepSeek", f"请求生成响应，提示词: '{prompt}'")

        # 异步包装chat函数
        chat_response = await asyncio.to_thread(self.chat, prompt)

        # 逐个产出响应块
        async for chunk in self._stream_chunks(chat_response):
            yield chunk

    async def _stream_chunks(self, response_generator: Generator) -> AsyncGenerator[str, None]:
        """将同步生成器转换为异步生成器"""
        for chunk in response_generator:
            if chunk.get('message', {}).get('content'):
                content = chunk['message']['content']
                yield content
                # 给事件循环一个让出的机会
                await asyncio.sleep(0)

    def chat(self, text, need_init=True):
        # 将用户输入添加到对话历史
        self.messages.append({
            'role': 'user',
            'content': text
        })

        log("DeepSeek", f"调用Ollama API，消息历史长度: {len(self.messages)}")

        # 使用完整的对话历史进行请求
        stream = ollama.chat(
            model='deepseek-r1:14b',
            messages=self.messages,
            stream=True
        )

        # 收集助手的回复
        assistant_message = {'role': 'assistant', 'content': ''}
        for chunk in stream:
            if chunk.get('message', {}).get('content'):
                content = chunk['message']['content']
                assistant_message['content'] += content
            yield chunk

        # 将助手的完整回复添加到对话历史
        self.messages.append(assistant_message)
        log("DeepSeek", f"完整回复添加到历史，当前长度: {len(self.messages)}")
        save(self.messages, "chat")

        if need_init:
            self._init_system()
            log("DeepSeek", "重置系统消息")


if __name__ == "__main__":
    pass