
"""
MiniQwen 推理脚本示例
"""

import argparse
import time
import torch
from transformers import AutoTokenizer, TextStreamer
from modeling_miniqwen import MiniQwenForCausalLM, MiniQwenConfig
from tokenization_miniqwen import MiniQwenTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="使用MiniQwen模型进行文本生成")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="模型路径，如果是None则会从零初始化模型"
    )
    parser.add_argument(
        "--tokenizer_path", 
        type=str, 
        default=None, 
        help="分词器路径，如果是None则会使用BERT的分词器"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Hello, I am MiniQwen, a lightweight language model. ", 
        help="提示文本"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100, 
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="生成温度"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="核采样top-p值"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50, 
        help="top-k采样值"
    )
    parser.add_argument(
        "--do_sample", 
        action="store_true", 
        help="是否进行采样"
    )
    parser.add_argument(
        "--streaming", 
        action="store_true", 
        help="是否流式输出"
    )
    return parser.parse_args()

def create_fresh_model():
    """
    创建一个全新的MiniQwen模型
    """
    print("创建新的MiniQwen模型...")
    # 配置模型参数
    config = MiniQwenConfig(
        vocab_size=30522,  # BERT词表大小
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=4,
        hidden_act="swiglu",
        max_position_embeddings=1024,
    )
    
    # 初始化模型
    model = MiniQwenForCausalLM(config)
    
    # 计算模型大小
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"模型大小: {model_size:.2f} MB")
    
    return model

def create_tokenizer(tokenizer_path=None):
    """
    创建或加载MiniQwen分词器
    """
    if tokenizer_path:
        print(f"从{tokenizer_path}加载分词器...")
        tokenizer = MiniQwenTokenizer.from_pretrained(tokenizer_path)
    else:
        print("创建基于BERT的分词器...")
        # 使用BERT分词器作为基础
        bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # 初始化我们的分词器
        tokenizer = MiniQwenTokenizer(
            vocab_file=None,  # 会自动从BERT加载词表
            bos_token="<s>",
            eos_token="</s>",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )
    
    return tokenizer

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建或加载模型
    if args.model_path:
        print(f"从{args.model_path}加载模型...")
        model = MiniQwenForCausalLM.from_pretrained(args.model_path)
    else:
        model = create_fresh_model()
    
    model.to(device)
    model.eval()
    
    # 创建或加载分词器
    tokenizer = create_tokenizer(args.tokenizer_path)
    
    # 处理输入
    prompt = args.prompt
    print(f"\n输入提示: {prompt}")
    
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 设置生成参数
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    # 流式输出设置
    if args.streaming:
        streamer = TextStreamer(tokenizer)
        gen_kwargs["streamer"] = streamer
    
    # 开始生成
    print("\n生成中...")
    start_time = time.time()
    
    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)
    
    # 计算生成时间
    end_time = time.time()
    generation_time = end_time - start_time
    tokens_generated = output_ids.shape[1] - input_ids.shape[1]
    tokens_per_second = tokens_generated / generation_time
    
    # 解码并打印结果（如果不是流式输出）
    if not args.streaming:
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"\n生成结果: {output_text}")
    
    print(f"\n生成统计:")
    print(f"- 生成时间: {generation_time:.2f}秒")
    print(f"- 生成token数: {tokens_generated}")
    print(f"- 生成速度: {tokens_per_second:.2f} tokens/秒")

if __name__ == "__main__":
    main()
