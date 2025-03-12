import os
import configparser
from datetime import datetime

def log(prefix, message):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{prefix}] {message}")

class APIManager:
    """
    API管理器，负责管理和提供各种API的配置信息
    """
    _instance = None
    
    # 单例模式实现
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(APIManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.config = configparser.ConfigParser()
        self.config_file = os.path.join(os.path.dirname(__file__), 'api_config.ini')
        self.example_file = os.path.join(os.path.dirname(__file__), 'api_config.example.ini')
        
        # 尝试加载配置文件
        self._load_config()
    
    def _load_config(self):
        """加载配置文件，如果不存在则创建一个默认配置"""
        try:
            if os.path.exists(self.config_file):
                self.config.read(self.config_file, encoding='utf-8')
                log("APIManager", f"已加载配置文件: {self.config_file}")
            else:
                log("APIManager", f"配置文件不存在: {self.config_file}")
                self._create_default_config()
        except Exception as e:
            log("APIManager", f"加载配置文件失败: {str(e)}")
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置文件"""
        log("APIManager", "创建默认配置...")
        
        # OpenAI API 配置
        self.config['OpenAI'] = {
            'api_url': 'https://api.openai.com/v1/',
            'api_key': '',
            'default_model': 'gpt-3.5-turbo',
            'temperature': '0.7',
            'max_tokens': '1000'
        }
        
        # DeepSeek API 配置
        self.config['DeepSeek'] = {
            'api_url': 'https://www.dmxapi.cn/v1/',
            'api_key': '',
            'default_model': 'claude-3-7-sonnet-20250219-thinking',
            'temperature': '0.7',
            'max_tokens': '2000'
        }
        
        # HuggingFace API 配置
        self.config['HuggingFace'] = {
            'api_url': 'https://api-inference.huggingface.co/models/',
            'token': '',
            'default_model': 'stabilityai/stable-diffusion-xl-base-1.0'
        }

        log("APIManager", "请复制example文件并重命名为api_config.ini，然后填入你的API密钥")

    def get_config(self, api_name, key=None, default=None):
        """获取指定API的配置项"""
        try:
            if api_name in self.config:
                if key:
                    return self.config[api_name].get(key, default)
                else:
                    return dict(self.config[api_name])
            else:
                log("APIManager", f"API配置不存在: {api_name}")
                return default
        except Exception as e:
            log("APIManager", f"获取配置失败: {str(e)}")
            return default
    
    def get_openai_config(self):
        """获取OpenAI API的配置"""
        return self.get_config('OpenAI')
    
    def get_deepseek_config(self):
        """获取DeepSeek API的配置"""
        return self.get_config('DeepSeek')
    
    def get_huggingface_config(self):
        """获取HuggingFace API的配置"""
        return self.get_config('HuggingFace')
    
    def save_config(self):
        """保存当前配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)
            log("APIManager", f"配置已保存: {self.config_file}")
            return True
        except Exception as e:
            log("APIManager", f"保存配置失败: {str(e)}")
            return False
    
    def update_config(self, api_name, key, value):
        """更新配置项"""
        try:
            if api_name not in self.config:
                self.config[api_name] = {}
            self.config[api_name][key] = value
            return True
        except Exception as e:
            log("APIManager", f"更新配置失败: {str(e)}")
            return False
            
# 创建全局实例
api_manager = APIManager()