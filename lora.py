#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from torchvision import transforms
from safetensors.torch import save_file

# Diffusers和相关优化库
from diffusers import (
    StableDiffusionXLPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection  # SDXL特有
)
from peft import (
    LoraConfig,
    get_peft_model
)
from accelerate import Accelerator
from accelerate.utils import set_seed

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ollama/lora_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LoRATrainer")


class SDXLLoRATrainer:
    def __init__(
            self,
            train_data_dir: str,
            output_dir: str,
            pretrained_model_name_or_path: str = "ollama/model_cache/models--John6666--wai-nsfw-illustrious-v110-sdxl/snapshots/5c5af76b9a8eeea210adc260bb4873b63a68c2ea",
            resolution: int = 1024,
            train_batch_size: int = 1,
            max_train_steps: int = 2000,
            learning_rate: float = 1e-5,
            lr_scheduler: str = "constant",
            lr_warmup_steps: int = 100,
            adam_beta1: float = 0.9,
            adam_beta2: float = 0.999,
            adam_weight_decay: float = 1e-2,
            adam_epsilon: float = 1e-8,
            seed: int = 42,
            lora_rank: int = 4,
            lora_alpha: int = 8,
            lora_dropout: float = 0.0,
            gradient_accumulation_steps: int = 4,
            mixed_precision: str = "bf16",  # bf16比fp16更稳定，但可以设为"fp16"节省显存
            checkpoint_frequency: int = 500,
            use_8bit_adam: bool = True,
            xformers_attention: bool = True,
            train_text_encoder: bool = True,
    ):
        """
        初始化SDXL LoRA训练器

        Args:
            train_data_dir: 训练数据目录
            output_dir: 输出目录
            pretrained_model_name_or_path: 预训练模型路径/名称
            resolution: 训练分辨率
            train_batch_size: 训练批次大小
            max_train_steps: 最大训练步数
            learning_rate: 学习率
            lr_scheduler: 学习率调度器类型
            lr_warmup_steps: 学习率预热步数
            adam_beta1: Adam β1参数
            adam_beta2: Adam β2参数
            adam_weight_decay: Adam权重衰减
            adam_epsilon: Adam ε参数
            seed: 随机种子
            lora_rank: LoRA秩
            lora_alpha: LoRA alpha值
            lora_dropout: LoRA dropout
            gradient_accumulation_steps: 梯度累积步数
            mixed_precision: 混合精度训练
            checkpoint_frequency: 检查点保存频率
            use_8bit_adam: 是否使用8bit Adam优化器
            xformers_attention: 是否使用xformers注意力机制
            train_text_encoder: 是否训练文本编码器
        """
        self.train_data_dir = Path(train_data_dir)
        self.output_dir = Path(output_dir)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.resolution = resolution
        self.train_batch_size = train_batch_size
        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_weight_decay = adam_weight_decay
        self.adam_epsilon = adam_epsilon
        self.seed = seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.checkpoint_frequency = checkpoint_frequency
        self.use_8bit_adam = use_8bit_adam
        self.xformers_attention = xformers_attention
        self.train_text_encoder = train_text_encoder

        # 确保输出目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 缓存目录
        self.cache_dir = Path(os.path.join(os.path.dirname(__file__), "model_cache"))

        # 设置设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            logger.warning("未检测到CUDA。训练将在CPU上进行，这将非常缓慢。")
            # 在CPU上自动减少训练规模
            self.max_train_steps = min(self.max_train_steps, 100)
            self.mixed_precision = "no"
            self.use_8bit_adam = False

        # 设置随机种子
        set_seed(self.seed)

    def prepare_dataset(self):
        """准备训练数据集"""
        logger.info(f"准备来自 {self.train_data_dir} 的数据集")

        # 图像转换
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # 加载图像文件
        image_files = list(self.train_data_dir.glob("*.jpg")) + list(self.train_data_dir.glob("*.png"))

        if not image_files:
            raise ValueError(f"在 {self.train_data_dir} 中未找到图像文件")

        logger.info(f"找到 {len(image_files)} 个训练图像")

        # 创建简单的训练数据集
        # 针对John6666/wai-nsfw-illustrious-v110-sdxl模型的提示词
        self.train_dataset = [(img_path, "masterpiece, best quality, highly detailed, 1girl") for img_path in
                              image_files]

        return self.train_dataset

    def load_models(self):
        """加载和准备模型组件"""
        logger.info(f"从本地路径加载模型: {self.pretrained_model_name_or_path}")

        # 加载SDXL模型的各个组件，使用低内存初始化
        # 针对12GB显存的优化: 使用torch.float16和low_cpu_mem_usage=True

        # 1. 加载文本编码器和分词器
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer",
        )

        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
        )

        # 2. 加载两个文本编码器
        text_encoder_dtype = torch.float16 if self.mixed_precision != "no" else torch.float32

        self.text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=text_encoder_dtype,
            low_cpu_mem_usage=True,
        )

        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            torch_dtype=text_encoder_dtype,
            low_cpu_mem_usage=True,
        )

        # 3. 加载VAE (使用本地的模型缓存中的VAE)
        vae_path = os.path.join(self.pretrained_model_name_or_path, "vae")
        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        # 4. 加载UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="unet",
            torch_dtype=torch.float16 if self.mixed_precision != "no" else torch.float32,
            low_cpu_mem_usage=True,
        )

        # 5. 加载噪声调度器
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder="scheduler",
        )

        # 6. 冻结VAE和文本编码器
        self.vae.requires_grad_(False)  # 冻结VAE权重

        # 设置为评估模式，节省显存
        self.vae.eval()
        if not self.train_text_encoder:
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            self.text_encoder.eval()
            self.text_encoder_2.eval()

        # 7. 对UNet应用LoRA
        unet_lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=[
                # 对SDXL UNet的适配器目标模块
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2"
            ],
        )

        # 创建LoRA UNet
        self.unet = get_peft_model(self.unet, unet_lora_config)

        # 8. 如果训练文本编码器，也应用LoRA
        if self.train_text_encoder:
            # 为第一个文本编码器配置LoRA
            text_encoder_lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "out_proj",
                    "fc1", "fc2"
                ],
            )

            # 为第二个文本编码器配置LoRA
            text_encoder_2_lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "out_proj",
                    "fc1", "fc2"
                ],
            )

            # 创建文本编码器LoRA
            self.text_encoder = get_peft_model(self.text_encoder, text_encoder_lora_config)
            self.text_encoder_2 = get_peft_model(self.text_encoder_2, text_encoder_2_lora_config)

        # 9. 为UNet和文本编码器准备VRAM优化
        if self.xformers_attention:
            try:
                import xformers
                self.unet.enable_xformers_memory_efficient_attention()
                if self.train_text_encoder:
                    self.text_encoder.gradient_checkpointing_enable()
                    self.text_encoder_2.gradient_checkpointing_enable()
                logger.info("已启用xformers内存优化")
            except ImportError:
                logger.warning("未安装xformers。对于更高效的训练，请安装xformers。")
                # 使用替代的内存优化方法
                try:
                    # 尝试使用set_attention_slice方法(较新版本的API)
                    self.unet.set_attention_slice(slice_size=1)
                    logger.info("已启用注意力切片 (新API)")
                except (AttributeError, TypeError):
                    # 如果失败，尝试直接设置注意力处理器
                    logger.info("使用基本内存优化")

        # 启用内存优化方法 - 使用梯度检查点
        self.unet.enable_gradient_checkpointing()
        logger.info("已启用梯度检查点")

        # 启用VAE分片以减少内存使用
        try:
            self.vae.enable_slicing()
            logger.info("已启用VAE切片")
        except AttributeError:
            logger.warning("当前VAE版本不支持切片")

        # 设置所有模型到设备
        self.vae.to(self.device, dtype=torch.float16)  # VAE总是使用fp16
        torch_dtype = torch.float16 if self.mixed_precision == "fp16" else (
            torch.bfloat16 if self.mixed_precision == "bf16" else torch.float32)
        self.unet.to(self.device, dtype=torch_dtype)
        self.text_encoder.to(self.device, dtype=torch_dtype)
        self.text_encoder_2.to(self.device, dtype=torch_dtype)

        logger.info(f"模型组件加载完成，使用精度: {self.mixed_precision}")

    def prepare_optimizer(self):
        """准备优化器和学习率调度器"""
        logger.info("准备优化器和学习率调度器")

        # 确定需要优化的参数
        if self.train_text_encoder:
            params_to_optimize = (
                    list(self.unet.parameters()) +
                    list(self.text_encoder.parameters()) +
                    list(self.text_encoder_2.parameters())
            )
        else:
            params_to_optimize = self.unet.parameters()

        # 创建优化器
        if self.use_8bit_adam:
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    params_to_optimize,
                    lr=self.learning_rate,
                    betas=(self.adam_beta1, self.adam_beta2),
                    weight_decay=self.adam_weight_decay,
                    eps=self.adam_epsilon,
                )
                logger.info("使用 8-bit Adam 优化器")
            except ImportError:
                logger.warning("未安装bitsandbytes。使用标准AdamW优化器，这将使用更多显存。")
                self.use_8bit_adam = False

        if not self.use_8bit_adam:
            self.optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=self.learning_rate,
                betas=(self.adam_beta1, self.adam_beta2),
                weight_decay=self.adam_weight_decay,
                eps=self.adam_epsilon,
            )

        # Prepare accelerator
        accelerator_project_config = {
            "logging_dir": os.path.join(self.output_dir, "accelerator_logs"),
        }

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision
            # project_config=accelerator_project_config,
        )

        # 设置学习率调度器
        if self.lr_scheduler == "constant":
            self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0
            )
        elif self.lr_scheduler == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_train_steps - self.lr_warmup_steps
            )
        elif self.lr_scheduler == "linear":
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.max_train_steps - self.lr_warmup_steps
            )
        else:
            raise ValueError(f"未知的学习率调度器: {self.lr_scheduler}")

        # 使用Accelerator准备所有训练组件
        if self.train_text_encoder:
            (
                self.unet,
                self.text_encoder,
                self.text_encoder_2,
                self.optimizer,
                self.lr_scheduler
            ) = self.accelerator.prepare(
                self.unet,
                self.text_encoder,
                self.text_encoder_2,
                self.optimizer,
                self.lr_scheduler
            )
        else:
            (
                self.unet,
                self.optimizer,
                self.lr_scheduler
            ) = self.accelerator.prepare(
                self.unet,
                self.optimizer,
                self.lr_scheduler
            )

        logger.info("优化器和学习率调度器准备完成")

    def encode_prompt(self, prompt, is_train=True):
        """编码提示词，生成文本嵌入"""
        # SDXL使用两个文本编码器
        # tokenizer_1/text_encoder处理标准提示词
        # tokenizer_2/text_encoder_2处理refiner提示词

        # Tokenize prompt
        text_inputs_1 = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_1.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_inputs_2 = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Move to device
        text_inputs_1 = text_inputs_1.to(self.device)
        text_inputs_2 = text_inputs_2.to(self.device)

        if not is_train:
            # 直接返回嵌入
            with torch.no_grad():
                prompt_embeds = self.text_encoder(text_inputs_1.input_ids)[0]
                pooled_prompt_embeds = self.text_encoder_2(text_inputs_2.input_ids)[0]

            return prompt_embeds, pooled_prompt_embeds

        # 在训练模式下返回输入
        return text_inputs_1, text_inputs_2

    def train_step(self, batch):
        """执行单个训练步骤"""
        image_path, prompt = batch

        # 加载和预处理图像
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transforms(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device, dtype=torch.float16)

        # 使用VAE编码图像
        latents = self.encode_image(image_tensor)

        # 添加噪声到潜在表示
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,), device=self.device)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # 准备文本输入
        text_inputs_1, text_inputs_2 = self.encode_prompt(prompt)

        # 前向传播通过UNet
        with self.accelerator.accumulate(self.unet):
            # 获取文本嵌入
            with torch.set_grad_enabled(self.train_text_encoder):
                prompt_embeds = self.text_encoder(text_inputs_1.input_ids)[0]
                pooled_prompt_embeds = self.text_encoder_2(text_inputs_2.input_ids)[0]

            # SDXL需要正确的添加条件嵌入
            # 修正: 正确准备SDXL的时间ID和条件
            add_text_embeds = pooled_prompt_embeds

            # 修正: 正确设置time_ids，SDXL使用的形状是[batch_size, 6]
            # 6个值代表: [height, width, 0, 0, crop_top, crop_left]
            original_size = (1024, 1024)  # 原始图像尺寸
            target_size = (self.resolution, self.resolution)  # 目标图像尺寸
            crops_coords_top_left = (0, 0)  # 裁剪起始点

            add_time_ids = torch.tensor(
                [
                    original_size[0],  # 原始高度
                    original_size[1],  # 原始宽度
                    crops_coords_top_left[0],  # 裁剪起点y
                    crops_coords_top_left[1],  # 裁剪起点x
                    target_size[0],  # 目标高度
                    target_size[1],  # 目标宽度
                ],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)

            logger.debug(f"Shapes: noisy_latents={noisy_latents.shape}, "
                         f"prompt_embeds={prompt_embeds.shape}, "
                         f"add_text_embeds={add_text_embeds.shape}, "
                         f"add_time_ids={add_time_ids.shape}")

            # UNet前向传播
            model_pred = self.unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids
                }
            ).sample

            # 计算损失
            loss = torch.nn.functional.mse_loss(model_pred, noise, reduction="mean")

            # 反向传播
            self.accelerator.backward(loss)

            # 更新优化器
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss.item()

    def encode_image(self, image_tensor):
        """使用VAE编码图像"""
        # VAE分片来节省内存
        with torch.no_grad():
            latents = self.vae.encode(image_tensor).latent_dist.sample() * 0.18215
        return latents

    def train(self):
        """执行完整的训练循环"""
        logger.info(f"开始训练，最大步数: {self.max_train_steps}")

        # 准备数据集和模型
        self.prepare_dataset()
        self.load_models()
        self.prepare_optimizer()

        # 显示内存使用信息
        if self.device == "cuda":
            logger.info(f"初始GPU内存使用: {torch.cuda.memory_allocated() / 1024 ** 2:.2f}MB")

        # 训练循环
        global_step = 0
        dataset_length = len(self.train_dataset)
        losses = []

        progress_bar = tqdm(range(self.max_train_steps), desc="训练进度")

        # 启用模型的训练模式
        self.unet.train()
        if self.train_text_encoder:
            self.text_encoder.train()
            self.text_encoder_2.train()

        # 训练循环
        while global_step < self.max_train_steps:
            for batch_idx in range(0, dataset_length, self.train_batch_size):
                # 检查是否已达到最大步数
                if global_step >= self.max_train_steps:
                    break

                # 获取批次数据
                batch = self.train_dataset[batch_idx % dataset_length]

                # 训练步骤
                loss = self.train_step(batch)
                losses.append(loss)

                # 更新进度条
                if global_step % 10 == 0:
                    avg_loss = sum(losses[-10:]) / min(len(losses), 10)
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    if self.device == "cuda":
                        current_memory = torch.cuda.memory_allocated() / 1024 ** 2
                        progress_bar.set_description(
                            f"步骤: {global_step}/{self.max_train_steps}, "
                            f"损失: {avg_loss:.4f}, "
                            f"GPU内存: {current_memory:.2f}MB"
                        )
                    else:
                        progress_bar.set_description(
                            f"步骤: {global_step}/{self.max_train_steps}, 损失: {avg_loss:.4f}"
                        )

                # 保存检查点
                if (global_step + 1) % self.checkpoint_frequency == 0 or global_step == self.max_train_steps - 1:
                    self.save_checkpoint(global_step + 1)

                # 更新全局步数
                global_step += 1
                progress_bar.update(1)

                # 清理内存
                if global_step % 50 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()

        # 训练结束，保存最终模型
        self.save_checkpoint("final")

        logger.info(f"训练完成! 共执行 {global_step} 步")

    def save_checkpoint(self, step):
        """保存模型检查点"""
        logger.info(f"保存步骤 {step} 的检查点...")

        # 创建检查点目录
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 确保模型在CPU上保存以避免内存问题
        if self.device != "cpu":
            self.accelerator.wait_for_everyone()

        # 获取未包装的模型
        unet = self.accelerator.unwrap_model(self.unet)

        # 收集LoRA权重
        lora_state_dict = {}

        # 添加UNet LoRA权重
        for key, value in unet.state_dict().items():
            if "lora" in key:
                lora_state_dict[f"unet.{key}"] = value.to("cpu")

        # 添加文本编码器LoRA权重(如果有)
        if self.train_text_encoder:
            text_encoder = self.accelerator.unwrap_model(self.text_encoder)
            text_encoder_2 = self.accelerator.unwrap_model(self.text_encoder_2)

            for key, value in text_encoder.state_dict().items():
                if "lora" in key:
                    lora_state_dict[f"text_encoder.{key}"] = value.to("cpu")

            for key, value in text_encoder_2.state_dict().items():
                if "lora" in key:
                    lora_state_dict[f"text_encoder_2.{key}"] = value.to("cpu")

        # 保存为safetensors格式
        safetensors_path = checkpoint_dir / "lora_weights.safetensors"
        save_file(lora_state_dict, safetensors_path)

        # 保存训练参数
        with open(checkpoint_dir / "training_config.txt", "w") as f:
            # 记录所有训练参数
            f.write(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"基础模型: {self.pretrained_model_name_or_path}\n")
            f.write(f"训练步数: {step}\n")
            f.write(f"学习率: {self.learning_rate}\n")
            f.write(f"训练分辨率: {self.resolution}\n")
            f.write(f"LoRA 秩: {self.lora_rank}\n")
            f.write(f"LoRA Alpha: {self.lora_alpha}\n")
            f.write(f"批次大小: {self.train_batch_size}\n")
            f.write(f"梯度累积步数: {self.gradient_accumulation_steps}\n")
            f.write(f"混合精度: {self.mixed_precision}\n")
            f.write(f"训练文本编码器: {self.train_text_encoder}\n")

        logger.info(f"检查点已保存到 {checkpoint_dir}")

        # 创建到lora文件夹的软链接
        if step == "final":
            # 复制最终权重到lora文件夹
            lora_dir = Path(os.path.dirname(__file__)) / "lora"
            lora_dir.mkdir(exist_ok=True)

            # 创建一个带有时间戳的文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            target_path = lora_dir / f"sdxl_lora_{timestamp}.safetensors"

            # 复制文件
            import shutil
            shutil.copy2(safetensors_path, target_path)
            logger.info(f"最终LoRA权重已复制到 {target_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练SDXL LoRA模型")

    # 基本参数
    parser.add_argument("--train_data_dir", type=str, default="ollama/train_processed",
                        help="训练数据目录")
    parser.add_argument("--output_dir", type=str, default="ollama/lora_checkpoints",
                        help="输出目录")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="ollama/model_cache/models--John6666--wai-nsfw-illustrious-v110-sdxl/snapshots/5c5af76b9a8eeea210adc260bb4873b63a68c2ea",
                        help="预训练模型路径/名称")

    # 训练参数
    parser.add_argument("--resolution", type=int, default=1024,
                        help="训练分辨率")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="训练批次大小")
    parser.add_argument("--max_train_steps", type=int, default=1000,
                        help="最大训练步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["constant", "cosine", "linear"],
                        help="学习率调度器类型")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="学习率预热步数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # LoRA参数
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha值")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout")

    # 优化参数
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="混合精度训练")
    parser.add_argument("--use_8bit_adam", action="store_true",
                        help="使用8bit Adam优化器")
    parser.add_argument("--xformers_attention", action="store_true",
                        help="使用xformers注意力机制")
    parser.add_argument("--train_text_encoder", action="store_true",
                        help="是否训练文本编码器")
    parser.add_argument("--checkpoint_frequency", type=int, default=500,
                        help="检查点保存频率")

    args = parser.parse_args()

    # 为12GB显存优化默认参数
    # 如果有更小的显存，可以进一步减小这些值
    if torch.cuda.is_available():
        cuda_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        # 自动调整参数，根据可用显存
        if cuda_memory_gb < 10:  # 8GB卡
            args.resolution = 768  # 降低分辨率
            args.train_batch_size = 1
            args.gradient_accumulation_steps = 8  # 增加梯度累积
            args.mixed_precision = "fp16"  # 使用fp16
            args.lora_rank = 4  # 降低LoRA秩
            args.train_text_encoder = False  # 不训练文本编码器
            args.use_8bit_adam = True
        elif cuda_memory_gb < 14:  # 12GB卡 - 默认显存大小
            args.resolution = 1024
            args.train_batch_size = 1
            args.gradient_accumulation_steps = 4
            args.mixed_precision = "fp16"
            args.lora_rank = 8
            args.train_text_encoder = True  # 可以训练文本编码器
            args.use_8bit_adam = True
        else:  # 16GB或更大
            # 使用默认参数
            pass

        logger.info(f"检测到GPU显存: {cuda_memory_gb:.2f}GB，已自动调整参数")

    # 创建并运行训练器
    trainer = SDXLLoRATrainer(
        train_data_dir=args.train_data_dir,
        output_dir=args.output_dir,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        max_train_steps=args.max_train_steps,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        checkpoint_frequency=args.checkpoint_frequency,
        use_8bit_adam=args.use_8bit_adam,
        xformers_attention=args.xformers_attention,
        train_text_encoder=args.train_text_encoder,
    )

    # 输出训练配置
    logger.info("====== 训练配置 ======")
    logger.info(f"数据目录: {args.train_data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"预训练模型: {args.pretrained_model_name_or_path}")
    logger.info(f"训练分辨率: {args.resolution}x{args.resolution}")
    logger.info(f"批次大小: {args.train_batch_size}")
    logger.info(f"梯度累积步数: {args.gradient_accumulation_steps}")
    logger.info(f"有效批次大小: {args.train_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"最大步数: {args.max_train_steps}")
    logger.info(f"学习率: {args.learning_rate}")
    logger.info(f"LoRA秩: {args.lora_rank}")
    logger.info(f"LoRA Alpha: {args.lora_alpha}")
    logger.info(f"混合精度: {args.mixed_precision}")
    logger.info(f"训练文本编码器: {args.train_text_encoder}")
    logger.info(f"使用8bit Adam: {args.use_8bit_adam}")
    logger.info("======================")

    # 执行训练
    trainer.train()

    logger.info("训练完成!")


if __name__ == "__main__":
    main()