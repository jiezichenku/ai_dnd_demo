import asyncio
import random

from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from PIL import Image
import os
from safetensors.torch import load_file

from api_manager import api_manager

class StableDiffusion:
    def __init__(self):
        # 设置模型缓存目录
        self.cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # 初始化模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)
        self.model = None
        self.is_sdxl = False
        self.sd_seed = random.randint(0, 120901813)
        # 获取HuggingFace配置
        hf_config = api_manager.get_huggingface_config()
        self.api_url = hf_config.get('api_url', 'https://api-inference.huggingface.co/models/')
        self.token = hf_config.get('token', '')
        self.default_model = hf_config.get('default_model', 'John6666/wai-nsfw-illustrious-v110-sdxl')
        # 预定义高质量动漫风格模型
        self.ANIME_MODELS = {
            "anything_v5": "stablediffusionapi/anything-v5",  # 类NAI风格
            "sdxl": "John6666/wai-nsfw-illustrious-v110-sdxl",  # 新型高质量动漫模型
        }

        if not self.load_model():
            print("Load model failed")
            return

    def load_model(self, model_id="sdxl", clip_skip=2):
        """
        加载 Stable Diffusion 模型
        Args:
            model_id: 模型ID，默认使用sdxl
        """
        if model_id:
            selected_model = model_id
        else:
            selected_model = "default"

        try:
            self.is_sdxl = "sdxl" in selected_model.lower()
            print(f"Loading model {model_id} on {self.device}...")

            if self.is_sdxl:
                from diffusers import StableDiffusionXLPipeline
                torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                with torch.amp.autocast("cuda"):
                    self.model = StableDiffusionXLPipeline.from_pretrained(
                        self.ANIME_MODELS[selected_model],
                        cache_dir=self.cache_dir,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                    )

                # 加载成功后转换为float16（如果使用CUDA）
                if self.device == "cuda":
                    self.model = self.model.to(torch.float16)
                    self.model.unet = self.model.unet.to(memory_format=torch.channels_last)
                    if hasattr(self.model, 'text_encoder'):
                        self.model.text_encoder = self.model.text_encoder.to(memory_format=torch.channels_last)
                    if hasattr(self.model, 'text_encoder_2'):
                        self.model.text_encoder_2 = self.model.text_encoder_2.to(memory_format=torch.channels_last)

                self.model.enable_attention_slicing()

                # 设置Clip Skip
                if clip_skip > 1:
                    # SDXL使用两个CLIP模型
                    # 设置第一个CLIP模型（text_encoder）的penultimate layer
                    self.model.text_encoder.text_model.final_layer_norm = self.model.text_encoder.text_model.encoder.layers[-(clip_skip)].layer_norm2
                    self.model.text_encoder.text_model.last_hidden_state = self.model.text_encoder.text_model.encoder.layers[-(clip_skip)].mlp

                    # 设置第二个CLIP模型（text_encoder_2）的penultimate layer
                    self.model.text_encoder_2.text_model.final_layer_norm = self.model.text_encoder_2.text_model.encoder.layers[-(clip_skip)].layer_norm2
                    self.model.text_encoder_2.text_model.last_hidden_state = self.model.text_encoder_2.text_model.encoder.layers[-(clip_skip)].mlp

                    print(f"Clip Skip set to {clip_skip}")
            else:
                self.model = StableDiffusionPipeline.from_pretrained(
                    self.ANIME_MODELS[selected_model],
                    cache_dir=self.cache_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                )

            self.model = self.model.to(self.device)

            # 可选：启用内存优化
            if self.device == "cuda":
                self.model.enable_attention_slicing()

            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def load_lora(self, lora_path, alpha=0.75):
        """
        加载 LoRA 模型
        Args:
            lora_path: LoRA 模型文件路径
            alpha: LoRA 权重，范围 0-1
        """
        if not self.model:
            raise RuntimeError("Base model must be loaded before loading LoRA")

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")

        try:
            # SDXL LoRA 加载逻辑
            if self.is_sdxl:
                # 使用 from_pretrained 方式加载 LoRA
                self.model.load_lora_weights(
                    lora_path,
                    weight_name="default",
                    adapter_name="default",
                    from_safetensors=True
                )

                # 设置 LoRA scale
                self.model.set_adapters(
                    adapter_names=["default"],
                    adapter_weights=[alpha]
                )

                # 确保模型在正确的设备上
                self.model.to(self.device)
                print(f"LoRA model loaded from {lora_path} with alpha={alpha}")
            else:
                raise NotImplementedError("LoRA loading for non-SDXL models not implemented")

        except Exception as e:
            print(f"Error loading LoRA: {str(e)}")
            raise

    def unload_lora(self):
        """
        卸载当前加载的 LoRA 模型
        """
        if not self.model:
            return

        if self.is_sdxl:
            self.model.unload_lora_weights()
            print("LoRA weights unloaded")

    def generate_image(self,
                       prompt,
                       negative_prompt="",
                       num_images=1,
                       width=1024,
                       height=1024,
                       num_inference_steps=20,
                       guidance_scale=4.5,
                       seed=None,
                       tiling=False,
                       clip_skip=2):
        """
        生成图像
        Args:
            prompt: 正向提示词
            negative_prompt: 负向提示词
            num_images: 生成图片数量
            width: 图片宽度
            height: 图片高度
            num_inference_steps: 推理步数
            guidance_scale: 提示词引导系数
            seed: 随机种子
        Returns:
            生成的图片列表
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()  # 清理GPU缓存

        # 优化批处理
        if num_images > 1:
            torch.set_grad_enabled(False)
            torch.backends.cudnn.benchmark = True

        default_anime_positive = """
                    masterpiece, best quality, 
                    1girl, solo, (full body:1.4),
                    professional, highly detailed,
                    pure white background, solid white
                    character design, game cg,
                """
        default_anime_negative = """
                    (worst quality, low quality, normal quality, cropped, watermark, text:1.4),
                    (realistic, photorealistic, semi-realistic:1.4),
                    (detailed background:1.4),
                    (gradient background:1.4),
                    (complex shading:1.4),
                    (dark shadows, heavy shading:1.4),
                    (painterly:1.4),
                    (sketch, sketchy:1.4),
                    (watercolor:1.4),
                    (noise, grain:1.4),
                    (blurry:1.4),
                    (depth of field, bokeh:1.4),
                    (bloom, glow effects:1.4),
                    (volumetric lighting:1.4),
                    (textured:1.4),
                    bad anatomy, bad hands,
                    multiple views, extra limbs,
                    ugly, deformed,
                    poorly drawn, low resolution,
                    blurry, jpeg artifacts,
                    username, signature,
                    extra details, fewer details,
                    oversaturated
                """
        if self.model is None:
            raise RuntimeError("Model not loaded. Please call load_model() first.")

        try:
            # 设置随机种子
            torch.manual_seed(self.sd_seed)

            # 生成图像
            full_positive = prompt + default_anime_positive
            print(f"正面提示词：{full_positive}")
            full_negative = negative_prompt + default_anime_negative
            print(f"负面提示词：{full_negative}")
            with torch.inference_mode(), torch.amp.autocast("cuda"):
                output = self.model(
                    prompt=prompt + default_anime_positive,
                    negative_prompt=negative_prompt + default_anime_negative,
                    num_images_per_prompt=num_images,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.manual_seed(seed) if seed is not None else None,
                )

                # 清理VRAM
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            # 获取生成的图片
            images = output.images

            # 返回生成的图片列表
            return images

        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

    def save_images(self, images, output_dir="outputs", base_filename="generated", start_index=0):
        """
        保存生成的图片
        Args:
            images: 图片列表
            output_dir: 输出目录
            base_filename: 基础文件名
            start_index: 输出图片初始index
        Returns:
            保存的文件路径列表
        """
        if not images:
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for i, image in enumerate(images):
            # 生成文件路径
            filename = f"{base_filename}_{i + start_index}.png"
            filepath = os.path.join(output_dir, filename)

            # 保存图片
            image.save(filepath)
            saved_paths.append(filepath)

        return saved_paths


def main(prompt="", lora=None):
    # 创建 StableDiffusion 实例
    sd = StableDiffusion()

    # sd.unload_lora()

    # 测试生成图片


    negative_prompt = "blurry, bad quality, distorted"

    if lora:
        sd.load_lora(lora)

    images = sd.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_images=2,
        height=768,
        width=1440
    )

    if images:
        # 保存生成的图片
        saved_paths = sd.save_images(images)
        print(f"Images saved to: {saved_paths}")
    else:
        print("Failed to generate images")

    sd.unload_lora()


def check_cuda_environment():
    """检查 CUDA 环境"""
    print("=== CUDA Environment Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

        # 检查 GPU 内存
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f}MB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1024 ** 2:.2f}MB")
    else:
        print("\nNo CUDA device available. Please check:")
        print("1. NVIDIA GPU is properly installed")
        print("2. NVIDIA drivers are up to date")
        print("3. CUDA Toolkit is installed")
        print("4. PyTorch is installed with CUDA support")

    print("===========================")

if __name__ == "__main__":
    # check_cuda_environment()
    prompt = """
        huohuo_(honkai_star_rail), white pantyhose, school uniform, twin tails, 
        shibari, bondage, red rope, suspension
    """
    lora = "lora/suspensionIllustrious.safetensors"
    main(prompt=prompt)
