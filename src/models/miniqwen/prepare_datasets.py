
"""
MiniQwen 数据准备脚本
用于下载和准备预训练和微调所需的数据集
"""

import os
import logging
import argparse
from datasets import load_dataset, concatenate_datasets

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def prepare_wikipedia(args):
    """
    准备Wikipedia数据集
    """
    logger.info(f"准备Wikipedia数据集 (版本: {args.wikipedia_version})...")
    try:
        # 尝试加载数据集
        if args.streaming:
            wiki_dataset = load_dataset("wikipedia", args.wikipedia_version, streaming=True)
            logger.info("Wikipedia数据集已加载（流式模式）")
            
            # 对流式数据集进行抽样以验证
            sample_count = 0
            for sample in wiki_dataset["train"].take(5):
                sample_count += 1
            logger.info(f"成功从Wikipedia流式数据集中抽取了{sample_count}个样本")
        else:
            # 只下载一部分数据进行验证
            wiki_dataset = load_dataset("wikipedia", args.wikipedia_version, split="train[:1%]")
            logger.info(f"Wikipedia样本示例（总计{len(wiki_dataset)}个）:")
            logger.info(f"- 标题: {wiki_dataset[0]['title']}")
            logger.info(f"- 文本片段: {wiki_dataset[0]['text'][:200]}...")
        
        logger.info("Wikipedia数据集准备完成")
        return True
    except Exception as e:
        logger.error(f"准备Wikipedia数据集时出错: {e}")
        return False

def prepare_arxiv(args):
    """
    准备arXiv论文数据集
    """
    logger.info("准备arXiv论文数据集...")
    try:
        # 尝试加载数据集
        if args.streaming:
            arxiv_dataset = load_dataset("arxiv_dataset", streaming=True)
            logger.info("arXiv数据集已加载（流式模式）")
            
            # 对流式数据集进行抽样以验证
            sample_count = 0
            for sample in arxiv_dataset["train"].take(5):
                sample_count += 1
            logger.info(f"成功从arXiv流式数据集中抽取了{sample_count}个样本")
        else:
            # 只下载一部分数据进行验证
            arxiv_dataset = load_dataset("arxiv_dataset", split="train[:1%]")
            logger.info(f"arXiv样本示例（总计{len(arxiv_dataset)}个）:")
            if len(arxiv_dataset) > 0:
                logger.info(f"- 标题: {arxiv_dataset[0]['title']}")
                logger.info(f"- 摘要片段: {arxiv_dataset[0]['abstract'][:200]}...")
        
        logger.info("arXiv数据集准备完成")
        return True
    except Exception as e:
        logger.error(f"准备arXiv数据集时出错: {e}")
        return False

def prepare_sciq(args):
    """
    准备SciQ数据集
    """
    logger.info("准备SciQ数据集...")
    try:
        # 尝试加载数据集
        sciq_dataset = load_dataset("sciq")
        
        # 打印数据集信息
        logger.info("SciQ数据集信息:")
        for split in sciq_dataset:
            logger.info(f"- {split}集: {len(sciq_dataset[split])}个样本")
        
        # 打印样本示例
        if "train" in sciq_dataset and len(sciq_dataset["train"]) > 0:
            example = sciq_dataset["train"][0]
            logger.info("\nSciQ样本示例:")
            logger.info(f"- 问题: {example['question']}")
            logger.info(f"- 支持文本: {example['support']}")
            logger.info(f"- 正确答案: {example['correct_answer']}")
            logger.info(f"- 干扰选项1: {example['distractor1']}")
            
        logger.info("SciQ数据集准备完成")
        return True
    except Exception as e:
        logger.error(f"准备SciQ数据集时出错: {e}")
        return False

def prepare_commonsenseqa(args):
    """
    准备CommonsenseQA数据集
    """
    logger.info("准备CommonsenseQA数据集...")
    try:
        # 尝试加载数据集
        commonsenseqa_dataset = load_dataset("commonsenseqa")
        
        # 打印数据集信息
        logger.info("CommonsenseQA数据集信息:")
        for split in commonsenseqa_dataset:
            logger.info(f"- {split}集: {len(commonsenseqa_dataset[split])}个样本")
        
        # 打印样本示例
        if "train" in commonsenseqa_dataset and len(commonsenseqa_dataset["train"]) > 0:
            example = commonsenseqa_dataset["train"][0]
            logger.info("\nCommonsenseQA样本示例:")
            logger.info(f"- 问题: {example['question']}")
            logger.info(f"- 选项: {[choice['text'] for choice in example['choices']['text']]}")
            logger.info(f"- 正确答案: {example['answerKey']}")
            
        logger.info("CommonsenseQA数据集准备完成")
        return True
    except Exception as e:
        logger.error(f"准备CommonsenseQA数据集时出错: {e}")
        return False

def prepare_all_datasets(args):
    """
    准备所有数据集
    """
    results = {}
    
    if args.wikipedia:
        results["wikipedia"] = prepare_wikipedia(args)
    
    if args.arxiv:
        results["arxiv"] = prepare_arxiv(args)
    
    if args.sciq:
        results["sciq"] = prepare_sciq(args)
    
    if args.commonsenseqa:
        results["commonsenseqa"] = prepare_commonsenseqa(args)
    
    # 打印准备结果摘要
    logger.info("\n=============== 数据集准备摘要 ===============")
    for dataset, success in results.items():
        status = "成功" if success else "失败"
        logger.info(f"{dataset}: {status}")
    
    if all(results.values()):
        logger.info("\n所有数据集都已成功准备！")
    else:
        logger.warning("\n部分数据集准备失败，请检查错误信息!")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="准备MiniQwen训练所需的数据集")
    parser.add_argument("--wikipedia", action="store_true", help="准备Wikipedia数据集")
    parser.add_argument("--arxiv", action="store_true", help="准备arXiv数据集")
    parser.add_argument("--sciq", action="store_true", help="准备SciQ数据集")
    parser.add_argument("--commonsenseqa", action="store_true", help="准备CommonsenseQA数据集")
    parser.add_argument("--all", action="store_true", help="准备所有数据集")
    parser.add_argument("--wikipedia_version", type=str, default="20220301.en", help="Wikipedia数据集版本")
    parser.add_argument("--streaming", action="store_true", help="使用流式模式加载大型数据集")
    parser.add_argument("--cache_dir", type=str, default=None, help="数据集缓存目录")
    
    args = parser.parse_args()
    
    # 如果选择了--all，则准备所有数据集
    if args.all:
        args.wikipedia = True
        args.arxiv = True
        args.sciq = True
        args.commonsenseqa = True
    
    # 如果没有选择任何数据集，则默认准备所有数据集
    if not any([args.wikipedia, args.arxiv, args.sciq, args.commonsenseqa]):
        logger.info("未指定要准备的数据集，默认准备所有数据集")
        args.wikipedia = True
        args.arxiv = True
        args.sciq = True
        args.commonsenseqa = True
    
    prepare_all_datasets(args)

if __name__ == "__main__":
    main()
