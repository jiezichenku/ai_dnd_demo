import sys
import asyncio
import time
from datetime import datetime

import torch
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLineEdit, QPushButton, QLabel, QSplitter,
                             QGroupBox, QGridLayout)
from PyQt6.QtCore import QObject, pyqtSignal as Signal, QThreadPool, Qt
from PyQt6.QtGui import QPixmap, QImage

from deepseek import Deepseek
from stable_diffusion import StableDiffusion
from character import Character


# 调试日志函数
def log(prefix, message):
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{prefix}] {message}")

# AI管理器
class AIManager(QObject):
    text_chunk_ready = Signal(str)  # 发送文本片段到UI
    image_ready = Signal(list)  # 发送生成的图像到UI
    thinking_changed = Signal(bool)  # 指示AI是否在思考
    prompt_extracted = Signal(str)  # 当提取到图像提示词时发射信号
    error_occurred = Signal(str)  # 错误信号

    def __init__(self, parent=None):
        super().__init__(parent)
        log("AIManager", "初始化AIManager")
        self.deepseek_service = Deepseek()
        self.sd_service = StableDiffusion()
        self.loop = None

        # 创建线程池
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)

        # 初始化任务集合和状态标志
        self.running_tasks = set()
        self._is_shutting_down = False
        self._cleanup_pending = False


        # 添加性能监控计数器
        self.conversation_count = 0
        self.last_memory_check = time.time()
        self.memory_check_interval = 1  # 每5次对话检查一次内存

        # 初始化内存监控
        self.monitor_memory()

    def monitor_memory(self):
        """监控内存使用情况"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            log("AIManager", f"GPU内存使用 - 已分配: {allocated:.2f}MB, 已预留: {reserved:.2f}MB")

        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        log("AIManager",
            f"CPU内存使用 - RSS: {memory_info.rss / 1024 ** 2:.2f}MB, VMS: {memory_info.vms / 1024 ** 2:.2f}MB")

    def cleanup(self):
        """清理资源"""
        log("AIManager", "开始清理资源")
        self._is_shutting_down = True

        # 取消所有运行中的任务
        for task in list(self.running_tasks):
            try:
                if hasattr(task, 'cancel'):
                    task.cancel()
            except Exception as e:
                log("AIManager", f"取消任务时出错: {str(e)}")

        # 等待线程池完成
        print(self.thread_pool)
        self.thread_pool.waitForDone()

        # 清理其他资源
        self.running_tasks.clear()
        if hasattr(self, 'sd_service'):
            del self.sd_service
        if hasattr(self, 'deepseek_service'):
            del self.deepseek_service

        log("AIManager", "资源清理完成")
        self._is_shutting_down = False

    def process_conversation(self, user_input: str):
        log("AIManager", f"开始处理对话，用户输入: '{user_input}'")
        # 在单独线程中启动处理流程
        self.thread_pool.start(lambda: self._process_in_thread(user_input))
        log("AIManager", "对话处理任务已加入线程池")

    def _process_in_thread(self, user_input: str):
        try:
            log("AIManager", f"线程启动，准备处理: '{user_input}'")
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # 设置默认的异常处理器
            def exception_handler(loop, context):
                exception = context.get('exception')
                msg = context.get('message')
                if exception:
                    log("AIManager", f"异步任务异常: {str(exception)}")
                    self.error_occurred.emit(str(exception))
                elif msg:
                    log("AIManager", f"异步任务消息: {msg}")

            self.loop.set_exception_handler(exception_handler)
            # 检查是否正在关闭
            if self._is_shutting_down:
                log("AIManager", "管理器正在关闭，取消处理")
                return

            log("AIManager", "开始运行主处理任务")
            main_task = self.loop.create_task(self._async_process(user_input))
            self.loop.run_until_complete(main_task)

            # finally:
            #     if not self._cleanup_pending:
            #         self._cleanup_pending = True
            #         self.cleanup()

        except Exception as e:
            log("AIManager", f"线程处理错误: {str(e)}")
            self.error_occurred.emit(f"处理错误: {str(e)}")

    async def _async_process(self, user_input: str):
        log("AIManager", f"开始异步处理用户输入: '{user_input}'")
        prompt_content = ""
        collecting_prompt = False
        collecting_think = False

        self.thinking_changed.emit(True)

        async for chunk in self.deepseek_service.generate_response(user_input):

            # 检查并处理</think>结束标签
            if collecting_think and "</think>" in chunk:
                parts = chunk.split("</think>")
                collecting_think = False
                # 如果标签后有内容，发送到UI
                if len(parts) > 1 and parts[1]:
                    self.text_chunk_ready.emit(parts[1])
                continue

            # 检查并处理<think>开始标签
            if not collecting_think and "<think>" in chunk:
                parts = chunk.split("<think>")
                # 发送标签之前的文本
                if parts[0]:
                    self.text_chunk_ready.emit(parts[0])
                collecting_think = True
                continue

            # 跳过思考内容
            if collecting_think:
                continue

            # 检查并处理}结束标签
            if collecting_prompt and "}" in chunk:
                parts = chunk.split("}")
                if parts[0]:
                    prompt_content += parts[0]

                log("AIManager", f"完整提示词: '{prompt_content}'")
                self.prompt_extracted.emit(prompt_content)

                # 启动图像生成任务
                log("AIManager", "创建图像生成协程任务")
                image_task = asyncio.create_task(self._generate_image(prompt_content))
                try:
                    # 等待图像生成任务完成
                    log("AIManager", "等待图像生成任务完成")
                    await image_task
                    log("AIManager", "图像生成任务完成")
                except Exception as e:
                    log("AIManager", f"图像生成任务出错: {str(e)}")
                    self.error_occurred.emit(f"图像生成失败: {str(e)}")

                collecting_prompt = False

                # 发送标签后的文本
                if len(parts) > 1 and parts[1]:
                    log("AIManager", "发送后的文本:")
                    log("AIManager", f"'{parts[1]}'")
                    self.text_chunk_ready.emit(parts[1])

                continue

            # 检查并处理{开始标签
            if not collecting_prompt and "{" in chunk:
                parts = chunk.split("{")
                if parts[0]:
                    self.text_chunk_ready.emit(parts[0])

                collecting_prompt = True
                prompt_content = ""

                # 如果标签后有内容，添加到提示词
                if len(parts) > 1 and parts[1]:
                    prompt_content += parts[1]

                continue

            # 收集提示词内容
            if collecting_prompt:
                prompt_content += chunk
            # 正常文本内容
            else:
                self.text_chunk_ready.emit(chunk)

        log("AIManager", "发出thinking_changed信号(False)")
        self.thinking_changed.emit(False)
        log("AIManager", "异步处理完成")

    async def _generate_image(self, prompt: str):
        log("AIManager", f"开始生成图像, 提示词: '{prompt}'")
        try:
            # 检查并清理内存
            self.conversation_count += 1
            if self.conversation_count % self.memory_check_interval == 0:
                self.monitor_memory()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            future = self.loop.create_future()
            log("AIManager", f"当前loop状态：{self.loop}, future: {future}")
            def generate_and_set_result():
                try:
                    result = self.sd_service.generate_image(prompt)
                    self.loop.call_soon_threadsafe(future.set_result, result)
                except Exception as e:
                    log("AIManager", f"执行generate_image报错{e}")
                    self.loop.call_soon_threadsafe(future.set_exception, e)

            self.thread_pool.start(generate_and_set_result)
            pil_images = await future
            log("AIManager", f"成功获取图像：{pil_images}")

            if not pil_images:
                log("AIManager", "生成图像失败，返回为空")
                self.error_occurred.emit("图像生成失败")
                return False
            else:
                log("AIManager", f"生成图像成功: {type(pil_images)}")

            # 转换PIL图像为QImage
            qt_images = []
            for pil_image in pil_images:
                try:
                    # 转换PIL图像为RGB模式
                    if pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')

                    # 获取图像数据
                    width = pil_image.width
                    height = pil_image.height
                    bytes_data = pil_image.tobytes('raw', 'RGB')

                    # 创建QImage并复制数据
                    qimage = QImage(bytes_data, width, height, width * 3, QImage.Format.Format_RGB888).copy()
                    qt_images.append(qimage)
                except Exception as e:
                    log("AIManager", f"图像转换错误: {str(e)}")
                    continue

            if qt_images:
                log("AIManager", f"图像生成完成，转换后的图像数量: {len(qt_images)}")
                # 确保在主线程发送信号
                self.image_ready.emit(qt_images)
                return True
            else:
                self.error_occurred.emit("图像转换失败")
                return False

        except Exception as e:
            log("AIManager", f"生成图像时出错: {str(e)}")
            import traceback
            log("AIManager", f"错误详情: {traceback.format_exc()}")
            self.error_occurred.emit(f"图像生成错误: {str(e)}")
            return False


# 主窗口
class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        log("ChatWindow", "初始化主窗口")
        self.setWindowTitle("DeepSeek Chat & Image Generator")
        self.setMinimumSize(800, 600)

        log("ChatWindow", "创建AIManager实例")
        self.ai_manager = AIManager(self)  # 设置父对象为主窗口
        log("ChatWindow", "连接AIManager信号")
        self.ai_manager.text_chunk_ready.connect(self.update_chat_text)
        self.ai_manager.image_ready.connect(self.update_image)
        self.ai_manager.thinking_changed.connect(self.set_thinking_status)
        self.ai_manager.prompt_extracted.connect(self.update_prompt_label)
        self.ai_manager.error_occurred.connect(self.handle_error)

        log("ChatWindow", "设置UI组件")
        self.setup_ui()
        log("ChatWindow", "主窗口初始化完成")

    def closeEvent(self, event):
        """窗口关闭时的处理"""
        log("ChatWindow", "窗口正在关闭")
        try:
            if hasattr(self, 'ai_manager'):
                self.ai_manager.cleanup()
        except Exception as e:
            log("ChatWindow", f"清理时出错: {str(e)}")
        super().closeEvent(event)

    def handle_error(self, error_message):
        """处理错误消息"""
        log("ChatWindow", f"收到错误: {error_message}")
        self.status_label.setText(f"错误: {error_message}")

    def __init__(self):
        super().__init__()
        log("ChatWindow", "初始化主窗口")
        self.setWindowTitle("DeepSeek Chat & Image Generator")
        self.setMinimumSize(800, 600)

        log("ChatWindow", "创建AIManager实例")
        self.ai_manager = AIManager(self)  # 设置父对象为主窗口
        log("ChatWindow", "连接AIManager信号")
        self.ai_manager.text_chunk_ready.connect(self.update_chat_text)
        self.ai_manager.image_ready.connect(self.update_image)
        self.ai_manager.thinking_changed.connect(self.set_thinking_status)
        self.ai_manager.prompt_extracted.connect(self.update_prompt_label)
        self.ai_manager.error_occurred.connect(self.handle_error)
        
        # 创建角色实例
        self.character = Character(1001, "冒险者", "探险家")
        
        log("ChatWindow", "设置UI组件")
        self.setup_ui()
        log("ChatWindow", "主窗口初始化完成")

    def setup_ui(self):
        log("ChatWindow", "开始设置UI")
        # 创建主窗口布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 创建左侧状态栏部分
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        
        # 角色基本信息组
        char_info_group = QGroupBox("角色信息")
        char_info_layout = QGridLayout()
        
        # 添加角色基本信息
        char_info_layout.addWidget(QLabel(f"姓名: {self.character.name}"), 0, 0)
        char_info_layout.addWidget(QLabel(f"职业: {self.character.role}"), 0, 1)
        char_info_layout.addWidget(QLabel(f"性别: {self.character.gender}"), 1, 0)
        char_info_layout.addWidget(QLabel(f"年龄: {self.character.age}"), 1, 1)
        char_info_layout.addWidget(QLabel(f"来自: {self.character.hometown}"), 2, 0, 1, 2)
        
        char_info_group.setLayout(char_info_layout)
        status_layout.addWidget(char_info_group)
        
        # 基础属性组
        base_attr_group = QGroupBox("基础属性")
        base_attr_layout = QGridLayout()
        
        # 添加基础属性
        base_attr_layout.addWidget(QLabel(f"力量(STR): {self.character.STR}"), 0, 0)
        base_attr_layout.addWidget(QLabel(f"体质(CON): {self.character.CON}"), 0, 1)
        base_attr_layout.addWidget(QLabel(f"体型(SIZ): {self.character.SIZ}"), 1, 0)
        base_attr_layout.addWidget(QLabel(f"敏捷(DEX): {self.character.DEX}"), 1, 1)
        base_attr_layout.addWidget(QLabel(f"外貌(APP): {self.character.APP}"), 2, 0)
        base_attr_layout.addWidget(QLabel(f"智力(INT): {self.character.INT}"), 2, 1)
        base_attr_layout.addWidget(QLabel(f"意志(POW): {self.character.POW}"), 3, 0)
        base_attr_layout.addWidget(QLabel(f"教育(EDU): {self.character.EDU}"), 3, 1)
        
        base_attr_group.setLayout(base_attr_layout)
        status_layout.addWidget(base_attr_group)
        
        # 派生属性组
        derived_attr_group = QGroupBox("派生属性")
        derived_attr_layout = QGridLayout()
        
        # 添加派生属性
        derived_attr_layout.addWidget(QLabel(f"移动力(MOV): {self.character.MOV}"), 0, 0)
        derived_attr_layout.addWidget(QLabel(f"生命值(HP): {self.character.HP}/{self.character.HP_max}"), 0, 1)
        derived_attr_layout.addWidget(QLabel(f"理智值(SAN): {self.character.SAN}/{self.character.SAN_max}"), 1, 0)
        derived_attr_layout.addWidget(QLabel(f"魔力值(MP): {self.character.MP}"), 1, 1)
        derived_attr_layout.addWidget(QLabel(f"幸运值(LUCK): {self.character.LUCK}"), 2, 0, 1, 2)
        
        derived_attr_group.setLayout(derived_attr_layout)
        status_layout.addWidget(derived_attr_group)
        
        # 添加填充空间
        status_layout.addStretch(1)

        # 创建中间聊天部分
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)

        # 聊天显示区域
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display)

        # 输入区域
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("输入您的消息...")
        self.input_field.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("发送")
        self.send_button.clicked.connect(self.send_message)

        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        chat_layout.addLayout(input_layout)

        # 创建右侧图像显示部分
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)

        image_label_header = QLabel("生成的图像")
        image_layout.addWidget(image_label_header)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        image_layout.addWidget(self.image_label)

        self.image_prompt_label = QLabel("等待图像生成...")
        self.image_prompt_label.setWordWrap(True)
        image_layout.addWidget(self.image_prompt_label)

        # 状态指示器
        self.status_label = QLabel("准备就绪")
        chat_layout.addWidget(self.status_label)

        # 使用分割器整合所有部分
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(status_widget)
        
        # 右侧聊天和图像分割
        right_splitter = QSplitter(Qt.Orientation.Horizontal)
        right_splitter.addWidget(chat_widget)
        right_splitter.addWidget(image_widget)
        right_splitter.setSizes([400, 400])
        
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([200, 600])  # 设置左边状态栏和右边内容的比例
        
        main_layout.addWidget(main_splitter)
        self.setCentralWidget(main_widget)

        # 初始欢迎消息
        log("ChatWindow", "添加欢迎消息")
        self.chat_display.append("初始化角色成功，请开始你的旅程。")
        log("ChatWindow", "UI设置完成")


    def send_message(self):
        user_input = self.input_field.text().strip()
        log("ChatWindow", f"发送消息函数调用，用户输入: '{user_input}'")

        if not user_input:
            log("ChatWindow", "用户输入为空，忽略请求")
            return

        self.input_field.clear()
        log("ChatWindow", "更新聊天显示 - 添加用户消息")
        self.chat_display.append(f"\n你: {user_input}\n")
        self.chat_display.append("DeepSeek: ")

        # 禁用输入区域直到回复完成
        log("ChatWindow", "禁用输入区域")
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)

        log("ChatWindow", "调用AIManager处理对话")
        self.ai_manager.process_conversation(user_input)

    def update_chat_text(self, text: str):
        self.chat_display.insertPlainText(text)
        # 自动滚动到底部
        self.chat_display.verticalScrollBar().setValue(
            self.chat_display.verticalScrollBar().maximum()
        )

    def update_image(self, image: list):
        image = image[0]
        log("ChatWindow", f"更新图像，尺寸: {image.width()}x{image.height()}")
        pixmap = QPixmap.fromImage(image)

        # 确保图像标签已经有几何信息
        if self.image_label.width() > 0 and self.image_label.height() > 0:
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        else:
            # 如果标签尚未调整大小，使用原始大小
            scaled_pixmap = pixmap

        self.image_label.setPixmap(scaled_pixmap)
        self.status_label.setText("图像生成完成")
        log("ChatWindow", "图像更新完成")


    def update_prompt_label(self, prompt: str):
        log("ChatWindow", f"更新图像提示词标签: '{prompt}'")
        self.image_prompt_label.setText(f"提示词: {prompt}")


    def set_thinking_status(self, is_thinking: bool):
        log("ChatWindow", f"设置思考状态: {is_thinking}")
        if is_thinking:
            self.status_label.setText("DeepSeek正在思考...")
        else:
            self.status_label.setText("回复完成")
            self.input_field.setEnabled(True)
            self.send_button.setEnabled(True)
        log("ChatWindow", "思考状态更新完成")


# 应用程序入口
def main():
    log("Main", "应用程序启动")
    app = QApplication(sys.argv)
    log("Main", "创建主窗口")
    window = ChatWindow()
    log("Main", "显示主窗口")
    window.show()
    log("Main", "进入应用程序主循环")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()