
"""
MiniQwen 高级优化脚本
提供更多高级选项来优化模型，包括量化、知识蒸馏和LoRA微调
"""

import os
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from modeling_miniqwen import MiniQwenForCausalLM, MiniQwenConfig
from tokenization_miniqwen import MiniQwenTokenizer

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    模型相关参数
    """
    model_path: str = field(
        metadata={"help": "MiniQwen模型的路径"}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "分词器的路径，如果与模型不同"}
    )
    teacher_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "用于知识蒸馏的教师模型路径（如有）"}
    )

@dataclass
class OptimizationArguments:
    """
    优化相关参数
    """
    optimization_method: str = field(
        default="none",
        metadata={"help": "优化方法，可选值为 'quantization', 'distillation', 'lora', 或 'none'"}
    )
    # 量化参数
    quantization_bits: int = field(
        default=8,
        metadata={"help": "量化位数，可选值为4或8"}
    )
    # LoRA参数
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA秩"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha参数"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout概率"}
    )
    # 知识蒸馏参数
    distillation_alpha: float = field(
        default=0.5,
        metadata={"help": "知识蒸馏损失权重"}
    )
    distillation_temperature: float = field(
        default=2.0,
        metadata={"help": "知识蒸馏温度"}
    )

def quantize_model(model, args):
    """
    对模型进行量化
    """
    if not HAS_BNB:
        raise ImportError("请安装bitsandbytes库以启用量化功能: pip install bitsandbytes")
    
    logger.info(f"将模型量化为{args.quantization_bits}位...")
    
    if args.quantization_bits == 8:
        # 8位量化
        model = bnb.nn.Linear8bitLt.replace_linear_layer(model)
        logger.info("已完成8位量化")
    elif args.quantization_bits == 4:
        # 4位量化
        model = bnb.nn.Linear4bit.replace_linear_layer(model)
        logger.info("已完成4位量化")
    else:
        raise ValueError(f"不支持的量化位数: {args.quantization_bits}，请选择4或8")
    
    return model

def setup_lora(model, args):
    """
    设置LoRA微调
    """
    if not HAS_PEFT:
        raise ImportError("请安装PEFT库以启用LoRA功能: pip install peft")
    
    logger.info("设置LoRA微调...")
    
    # 准备模型进行LoRA微调
    if args.quantization_bits in [4, 8]:
        model = prepare_model_for_kbit_training(model)
    
    # 创建LoRA配置
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印训练参数数量
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    logger.info(f"可训练参数: {trainable_params}, 总参数: {all_param}, 比例: {trainable_params/all_param:.2%}")
    
    return model

class DistillationTrainer(Trainer):
    """
    用于知识蒸馏的训练器
    """
    def __init__(self, teacher_model=None, distillation_alpha=0.5, distillation_temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distillation_alpha = distillation_alpha
        self.distillation_temperature = distillation_temperature
        
        # 确保教师模型处于评估模式
        if self.teacher_model is not None:
            self.teacher_model.eval()
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # 标准交叉熵损失
        outputs = model(**inputs)
        student_loss = outputs.loss
        
        # 如果没有教师模型，则只使用标准损失
        if self.teacher_model is None:
            return (student_loss, outputs) if return_outputs else student_loss
        
        # 知识蒸馏损失
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        # 提取logits
        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits
        
        # 应用温度缩放
        student_logits_t = student_logits / self.distillation_temperature
        teacher_logits_t = teacher_logits / self.distillation_temperature
        
        # 计算KL散度损失
        distillation_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits_t, dim=-1),
            torch.nn.functional.softmax(teacher_logits_t, dim=-1),
            reduction="batchmean",
        ) * (self.distillation_temperature ** 2)
        
        # 组合损失
        loss = (1 - self.distillation_alpha) * student_loss + self.distillation_alpha * distillation_loss
        
        return (loss, outputs) if return_outputs else loss

def load_model_and_tokenizer(model_args):
    """
    加载模型和分词器
    """
    logger.info(f"从 {model_args.model_path} 加载模型...")
    model = MiniQwenForCausalLM.from_pretrained(model_args.model_path)
    
    tokenizer_path = model_args.tokenizer_path or model_args.model_path
    logger.info(f"从 {tokenizer_path} 加载分词器...")
    tokenizer = MiniQwenTokenizer.from_pretrained(tokenizer_path)
    
    return model, tokenizer

def load_teacher_model(model_args):
    """
    加载教师模型用于蒸馏
    """
    if model_args.teacher_model_path:
        logger.info(f"从 {model_args.teacher_model_path} 加载教师模型...")
        teacher_model = MiniQwenForCausalLM.from_pretrained(model_args.teacher_model_path)
        return teacher_model
    return None

def save_optimized_model(model, tokenizer, output_dir, optimization_method):
    """
    保存优化后的模型
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"保存优化后的模型到 {output_dir}...")
    
    # 保存模型
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir)
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    
    # 保存优化方法信息
    with open(os.path.join(output_dir, "optimization_info.txt"), "w") as f:
        f.write(f"优化方法: {optimization_method}\n")
    
    logger.info("模型保存完成")

def benchmark_model(model, tokenizer, device, sequence_length=512, batch_size=1, num_iterations=10):
    """
    基准测试模型的推理性能
    """
    logger.info("开始性能基准测试...")
    
    # 生成随机输入
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, sequence_length), dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    
    # 预热
    logger.info("预热中...")
    with torch.no_grad():
        for _ in range(3):
            model(input_ids=input_ids, attention_mask=attention_mask)
    
    # 计时
    logger.info(f"运行{num_iterations}次迭代...")
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input_ids=input_ids, attention_mask=attention_mask)
    end_time.record()
    
    # 同步
    torch.cuda.synchronize()
    
    # 计算时间
    elapsed_time = start_time.elapsed_time(end_time) / 1000  # 转换为秒
    avg_time = elapsed_time / num_iterations
    
    logger.info(f"平均推理时间: {avg_time:.4f}秒/次 (批次大小={batch_size}, 序列长度={sequence_length})")
    
    # 计算内存占用
    memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    logger.info(f"GPU内存占用: {memory_allocated:.2f} MB")
    
    return {"avg_inference_time": avg_time, "memory_usage_mb": memory_allocated}

def main():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, OptimizationArguments, TrainingArguments))
    model_args, optimization_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_args)
    
    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 应用优化方法
    optimization_info = {"method": optimization_args.optimization_method}
    
    if optimization_args.optimization_method == "quantization":
        logger.info("应用量化优化...")
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        logger.info(f"原始模型大小: {original_size:.2f} MB")
        
        model = quantize_model(model, optimization_args)
        
        quantized_size = sum(p.numel() * (p.element_size() if hasattr(p, "element_size") else 4) for p in model.parameters()) / (1024 * 1024)
        logger.info(f"量化后模型大小: {quantized_size:.2f} MB")
        logger.info(f"大小减少: {original_size - quantized_size:.2f} MB ({(original_size - quantized_size) / original_size * 100:.2f}%)")
        
        optimization_info["quantization_bits"] = optimization_args.quantization_bits
        optimization_info["size_reduction_percentage"] = (original_size - quantized_size) / original_size * 100
    
    elif optimization_args.optimization_method == "lora":
        logger.info("应用LoRA优化...")
        model = setup_lora(model, optimization_args)
        
        optimization_info["lora_rank"] = optimization_args.lora_rank
        optimization_info["lora_alpha"] = optimization_args.lora_alpha
        optimization_info["lora_dropout"] = optimization_args.lora_dropout
    
    elif optimization_args.optimization_method == "distillation":
        logger.info("应用知识蒸馏优化...")
        teacher_model = load_teacher_model(model_args)
        if teacher_model is None:
            raise ValueError("进行知识蒸馏需要指定教师模型路径")
        
        teacher_model = teacher_model.to(device)
        
        optimization_info["distillation_alpha"] = optimization_args.distillation_alpha
        optimization_info["distillation_temperature"] = optimization_args.distillation_temperature
    
    # 运行基准测试（如果不是training）
    if not training_args.do_train and not training_args.do_eval:
        logger.info("未指定训练或评估操作，执行基准测试...")
        benchmark_results = benchmark_model(model, tokenizer, device)
        optimization_info.update(benchmark_results)
    
    # 设置训练器（如果需要）
    if training_args.do_train or training_args.do_eval:
        if optimization_args.optimization_method == "distillation":
            trainer = DistillationTrainer(
                teacher_model=teacher_model,
                distillation_alpha=optimization_args.distillation_alpha,
                distillation_temperature=optimization_args.distillation_temperature,
                model=model,
                args=training_args,
                tokenizer=tokenizer,
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                tokenizer=tokenizer,
            )
        
        # 训练和评估（如果需要）
        # 注意：这里没有加载数据集，因为这需要根据具体任务来确定
        # 在实际使用中，你需要加载数据集并传递给Trainer
        logger.warning("未提供训练或评估数据集，跳过训练和评估步骤。请在实际使用中加载合适的数据集。")
    
    # 保存优化后的模型
    output_dir = os.path.join(training_args.output_dir, f"miniqwen_{optimization_args.optimization_method}")
    save_optimized_model(model, tokenizer, output_dir, optimization_args.optimization_method)
    
    # 保存优化信息
    with open(os.path.join(output_dir, "optimization_details.txt"), "w") as f:
        for key, value in optimization_info.items():
            f.write(f"{key}: {value}\n")
    
    logger.info("优化过程完成!")

if __name__ == "__main__":
    main()
