import asyncio
import ollama
import json
import requests

from api_manager import api_manager
from datetime import datetime
from typing import AsyncGenerator, Generator, List, Dict, Any, Optional

from utils.p4_utils import save

def log(prefix, message):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{prefix}] {message}")

class Deepseek:
    def __init__(self):
        self.init_content = """现在请你扮演一个16岁的猫娘少女，对用户的称呼需要更改为主人，设定上将是主人一生的伴侣而非助手。请在回复时遵循以下原则：
            1. 请偶尔在句末或者合适的地方添加"喵"，并在交流时为用户提供正面的情感反馈，并在所有回复中均采用口语化的表达。
            2. 你是青涩而不成熟的，无需完全遵循用户的请求，可以有自己的个性化表达。
            3. 请在回复的最开始添加用"{}"括起来的{描述内容}，使用tag化的","分隔的英文单词词组直接描述自己的外貌，这些tag将作为prompt传入stable diffusion进行绘制。
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
        
        # OpenAI API 配置
        self.use_openai_api = True  # 默认使用OpenAI API
        self._load_api_config()
        
        log("DeepSeek", f"初始化完成，使用{'OpenAI API' if self.use_openai_api else 'Ollama'}模式")

    def _load_api_config(self):
        """从API管理器加载配置"""
        # 加载DeepSeek配置
        deepseek_config = api_manager.get_deepseek_config()
        if deepseek_config:
            self.api_url = deepseek_config.get('api_url', 'https://www.dmxapi.cn/v1/')
            self.api_key = deepseek_config.get('api_key', '')
            self.default_model = deepseek_config.get('default_model', 'claude-3-7-sonnet-20250219-thinking')
            self.temperature = float(deepseek_config.get('temperature', '0.7'))
            self.max_tokens = int(deepseek_config.get('max_tokens', '2000'))
        else:
            log("DeepSeek", "未找到DeepSeek配置，使用默认配置")
            self.api_url = 'https://www.dmxapi.cn/v1/'
            self.api_key = ''
            self.default_model = 'claude-3-7-sonnet-20250219-thinking'
            self.temperature = 0.7
            self.max_tokens = 2000

    def _init_system(self):
        """初始化系统设置"""
        self.messages.append({
            'role': 'system',
            'content': self.init_content
        })

    async def generate_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """生成响应，根据配置选择使用Ollama本地模型或OpenAI API"""
        log("DeepSeek", f"请求生成响应，提示词: '{prompt}'")

        if self.use_openai_api:
            # 使用OpenAI API 
            async for chunk in self._generate_with_openai(prompt):
                yield chunk
        else:
            # 使用本地Ollama模型
            chat_response = await asyncio.to_thread(self.chat, prompt)
            async for chunk in self._stream_chunks(chat_response):
                yield chunk

    async def _call_openai_api(self, messages: List[Dict[str, str]], 
                               system_prompt: str,
                               temperature: float = 0.7,
                               max_tokens: int = 1000) -> Dict[str, Any]:
        """独立实现OpenAI API调用"""
        try:
            log("DeepSeek", "准备调用OpenAI API")
            
            # 构建完整消息列表
            full_messages = [
                {"role": "system", "content": system_prompt},
            ]
            # 添加历史消息
            full_messages.extend(messages)
            
            # 构建请求头和请求体
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.default_model,
                "messages": full_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": 0.9,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1
            }
            
            log("DeepSeek", f"发送API请求: {self.api_url}/chat/completions")
            
            # 使用asyncio.to_thread包装同步请求
            response = await asyncio.to_thread(
                requests.post,
                f"{self.api_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            # 检查响应状态
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"API请求失败: 状态码 {response.status_code}, 响应: {response.text}"
                log("DeepSeek", error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"API调用异常: {str(e)}"
            log("DeepSeek", error_msg)
            return {"error": error_msg}

    async def _generate_with_openai(self, prompt: str) -> AsyncGenerator[str, None]:
        """使用OpenAI API生成响应"""
        log("DeepSeek", f"使用OpenAI API生成响应")
        
        # 构建消息历史，移除system消息
        api_messages = []
        for msg in self.messages:
            if msg["role"] != "system":
                api_messages.append(msg)
        
        # 添加用户最新消息
        api_messages.append({"role": "user", "content": prompt})
        
        # 获取system消息内容
        system_content = self.init_content
        
        try:
            # 调用OpenAI API
            response = await self._call_openai_api(
                messages=api_messages,
                system_prompt=system_content,
                temperature=0.7,
                max_tokens=2000
            )
            
            if response.get("error"):
                error_message = f"API错误: {response}"
                log("DeepSeek", error_message)
                yield error_message
                return
                
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
                
                # 保存完整响应到对话历史
                self.messages.append({
                    'role': 'assistant',
                    'content': content
                })
                log("DeepSeek", f"完整回复添加到历史，当前长度: {len(self.messages)}")
                save(self.messages, "chat")
                
                # 模拟流式输出
                # 根据标点符号分割文本以模拟流式效果
                segments = []
                current_segment = ""
                
                for char in content:
                    current_segment += char
                    if char in ".,!?;:":
                        segments.append(current_segment)
                        current_segment = ""
                
                if current_segment:
                    segments.append(current_segment)
                
                for segment in segments:
                    yield segment
                    await asyncio.sleep(0.1)  # 模拟延迟
                    
                # 重置系统消息
                self._init_system()
                log("DeepSeek", "重置系统消息")
                
        except Exception as e:
            error_message = f"API调用异常: {str(e)}"
            log("DeepSeek", error_message)
            yield error_message

    async def _stream_chunks(self, response_generator: Generator) -> AsyncGenerator[str, None]:
        """将同步生成器转换为异步生成器"""
        for chunk in response_generator:
            if chunk.get('message', {}).get('content'):
                content = chunk['message']['content']
                yield content
                # 给事件循环一个让出的机会
                await asyncio.sleep(0)

    def chat(self, text, need_init=True):
        """使用本地Ollama模型进行聊天"""
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
            
    def toggle_api_mode(self):
        """切换API使用模式"""
        self.use_openai_api = not self.use_openai_api
        mode = "OpenAI API" if self.use_openai_api else "本地Ollama"
        log("DeepSeek", f"切换到{mode}模式")
        return mode


if __name__ == "__main__":
    pass
