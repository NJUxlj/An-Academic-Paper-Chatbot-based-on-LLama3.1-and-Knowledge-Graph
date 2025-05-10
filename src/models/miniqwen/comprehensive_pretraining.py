
"""
Comprehensive MiniQwen Pretraining Script

This script performs continued pretraining of the MiniQwen model on multiple datasets:
- Books (BookCorpus)
- Code (The Stack)
- Reddit discussions (Pushshift Reddit)
- Academic papers (ArXiv and PubMed, filtered for post-2020)

Using Hugging Face's Trainer class with distributed training support.
"""

import os
import logging
import math
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import partial

import torch
import datasets
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict, interleave_datasets
import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint, speed_metrics
from transformers.utils import check_min_version, send_example_telemetry

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

from modeling_miniqwen import MiniQwenForCausalLM, MiniQwenConfig
from tokenization_miniqwen import MiniQwenTokenizer

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained MiniQwen model or model identifier from huggingface.co/models"}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer path. If not specified, will use model_path"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Allow loading remote code when loading a model from a remote location."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "This is useful in low-resource or server environments when you don't want to use a lot of CPU memory to load models you don't use."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset_config: Optional[str] = field(
        default="config.json",
        metadata={"help": "JSON file containing dataset configurations"}
    )
    max_seq_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(
        default=False, 
        metadata={"help": "Enable streaming mode for large datasets"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    dataset_sampling_rates: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of sampling rates for each dataset, must match the number and order of datasets"}
    )
    pre_2020_sampling_rate: float = field(
        default=0.2,
        metadata={"help": "Sampling rate for pre-2020 academic papers (to prioritize recent research)"}
    )

def create_fresh_model():
    """
    Create a new MiniQwen model
    """
    logger.info("Creating new MiniQwen model...")
    # Configure model parameters
    config = MiniQwenConfig(
        vocab_size=30522,  # BERT vocabulary size
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=6,
        num_attention_heads=12,
        num_key_value_heads=4,
        hidden_act="swiglu",
        max_position_embeddings=1024,
    )
    
    # Initialize model
    model = MiniQwenForCausalLM(config)
    
    # Calculate model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    logger.info(f"Model size: {model_size:.2f} MB")
    
    return model

def create_tokenizer(tokenizer_path=None):
    """
    Create or load MiniQwen tokenizer
    """
    if tokenizer_path:
        logger.info(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = MiniQwenTokenizer.from_pretrained(tokenizer_path)
    else:
        logger.info("Creating BERT-based tokenizer...")
        # Initialize our tokenizer
        tokenizer = MiniQwenTokenizer(
            vocab_file=None,  # Will automatically load BERT vocabulary
            bos_token="<s>",
            eos_token="</s>",
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )
    
    return tokenizer

def load_and_process_dataset(dataset_config, tokenizer, data_args, training_args):
    """
    Load and process datasets based on configuration
    """
    # Load dataset configuration
    with open(dataset_config, "r") as f:
        config = json.load(f)
    
    # Initialize datasets dictionary and dataset sampling weights
    raw_datasets = {}
    dataset_weights = []
    
    # Parse sampling rates if provided
    if data_args.dataset_sampling_rates:
        try:
            dataset_weights = [float(rate) for rate in data_args.dataset_sampling_rates.split(",")]
        except ValueError:
            logger.warning("Invalid dataset_sampling_rates format. Using default equal weights.")
            dataset_weights = []
    
    # Function to filter ArXiv and PubMed papers by year
    def filter_by_year(example, year_threshold=2020):
        # Check if 'year' or 'date' field exists and filter
        if 'year' in example:
            return example['year'] >= year_threshold
        elif 'date' in example:
            # Assuming date format as YYYY-MM-DD or similar
            try:
                year = int(example['date'][:4])
                return year >= year_threshold
            except (ValueError, TypeError, IndexError):
                # If year parsing fails, keep the example but with lower weight
                return True
        # If no date information, keep by default but will apply sampling later
        return True

    # Process each dataset specified in config
    for dataset_name, dataset_info in config["datasets"].items():
        logger.info(f"Loading dataset: {dataset_name}")
        
        try:
            # Handle dataset based on source
            if dataset_info["source"] == "huggingface":
                # Load from Hugging Face datasets
                if data_args.streaming:
                    dataset = load_dataset(
                        dataset_info["name"],
                        dataset_info.get("config", None),
                        streaming=True,
                        cache_dir=training_args.cache_dir
                    )
                else:
                    dataset = load_dataset(
                        dataset_info["name"],
                        dataset_info.get("config", None),
                        cache_dir=training_args.cache_dir
                    )
            elif dataset_info["source"] == "local":
                # Load from local files
                data_files = dataset_info.get("files", {})
                if data_args.streaming:
                    dataset = load_dataset(
                        "json", 
                        data_files=data_files,
                        streaming=True,
                        cache_dir=training_args.cache_dir
                    )
                else:
                    dataset = load_dataset(
                        "json", 
                        data_files=data_files,
                        cache_dir=training_args.cache_dir
                    )
            else:
                logger.warning(f"Unknown dataset source for {dataset_name}, skipping.")
                continue
            
            # Apply filters for academic papers to prioritize post-2020 content
            if dataset_name in ['arxiv', 'pubmed']:
                logger.info(f"Filtering {dataset_name} for papers after 2020")
                
                if data_args.streaming:
                    # For streaming datasets, we can filter on-the-fly
                    post_2020 = dataset.filter(lambda x: filter_by_year(x, 2020))
                    pre_2020_sampled = dataset.filter(
                        lambda x: not filter_by_year(x, 2020)
                    ).shuffle(seed=training_args.seed, buffer_size=10000).select(range(int(1000000 * data_args.pre_2020_sampling_rate)))
                    
                    # Combine filtered datasets
                    dataset = interleave_datasets([post_2020, pre_2020_sampled], probabilities=[0.8, 0.2])
                else:
                    # For in-memory datasets, filter and sample directly
                    splits = {}
                    for split in dataset:
                        # Split into pre and post 2020
                        post_2020_indices = [i for i, example in enumerate(dataset[split]) if filter_by_year(example, 2020)]
                        pre_2020_indices = [i for i, example in enumerate(dataset[split]) if not filter_by_year(example, 2020)]
                        
                        # Sample pre-2020 data
                        if pre_2020_indices:
                            import random
                            random.seed(training_args.seed)
                            pre_2020_sampled = random.sample(
                                pre_2020_indices, 
                                k=min(int(len(pre_2020_indices) * data_args.pre_2020_sampling_rate), len(pre_2020_indices))
                            )
                            selected_indices = post_2020_indices + pre_2020_sampled
                            splits[split] = dataset[split].select(selected_indices)
                        else:
                            splits[split] = dataset[split]
                    
                    dataset = DatasetDict(splits)
            
            # Apply data preprocessing
            for split in dataset:
                # Extract text field based on configuration
                text_column = dataset_info.get("text_column", "text")
                
                def extract_text(examples):
                    if text_column in examples:
                        return {"text": examples[text_column]}
                    else:
                        # If the specified text column doesn't exist, try to construct from other fields
                        texts = []
                        for i in range(len(examples.get(next(iter(examples)), []))):
                            parts = []
                            # For academic papers, combine title and abstract/body
                            if "title" in examples and i < len(examples["title"]):
                                parts.append(f"Title: {examples['title'][i]}")
                            if "abstract" in examples and i < len(examples["abstract"]):
                                parts.append(f"Abstract: {examples['abstract'][i]}")
                            if "body" in examples and i < len(examples["body"]):
                                parts.append(f"Body: {examples['body'][i]}")
                            
                            texts.append("\n\n".join(parts))
                        return {"text": texts}
                
                # Apply text extraction
                if data_args.streaming:
                    dataset[split] = dataset[split].map(extract_text)
                else:
                    dataset[split] = dataset[split].map(
                        extract_text,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=dataset[split].column_names,
                        desc=f"Extracting text from {dataset_name} {split}",
                    )
            
            # Add to raw_datasets
            raw_datasets[dataset_name] = dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
    
    # Combine datasets based on configuration mode
    if len(raw_datasets) == 0:
        raise ValueError("No datasets were successfully loaded.")
    
    logger.info(f"Loaded {len(raw_datasets)} datasets: {', '.join(raw_datasets.keys())}")
    
    # Determine dataset combination method
    if config.get("combination_mode", "concatenate") == "interleave" and data_args.streaming:
        # For streaming datasets, use interleave with probabilities
        
        # Ensure we have sampling weights for each dataset
        if not dataset_weights or len(dataset_weights) != len(raw_datasets):
            # Use equal weights if not specified
            dataset_weights = [1.0 / len(raw_datasets)] * len(raw_datasets)
        else:
            # Normalize weights to sum to 1
            total_weight = sum(dataset_weights)
            dataset_weights = [w / total_weight for w in dataset_weights]
        
        logger.info(f"Interleaving datasets with probabilities: {list(zip(raw_datasets.keys(), dataset_weights))}")
        
        # Create interleaved dataset
        dataset_list = [dataset["train"] for dataset in raw_datasets.values()]
        combined_train = interleave_datasets(dataset_list, probabilities=dataset_weights)
        combined_datasets = {"train": combined_train}
        
        # Create a small validation set if available
        if any("validation" in dataset for dataset in raw_datasets.values()):
            val_datasets = [dataset["validation"] for dataset in raw_datasets.values() if "validation" in dataset]
            if val_datasets:
                combined_datasets["validation"] = interleave_datasets(
                    val_datasets, 
                    probabilities=[1.0/len(val_datasets)] * len(val_datasets)
                )
        
    else:
        # For in-memory datasets or when concatenation is preferred
        logger.info("Concatenating datasets")
        
        # Combine training data
        train_datasets = []
        for dataset_name, dataset in raw_datasets.items():
            if "train" in dataset:
                train_datasets.append(dataset["train"])
        
        if train_datasets:
            combined_train = concatenate_datasets(train_datasets)
            
            # Create validation split if needed
            if not any("validation" in dataset for dataset in raw_datasets.values()):
                # Create a validation split
                train_valid = combined_train.train_test_split(
                    test_size=data_args.validation_split_percentage / 100,
                    seed=training_args.seed
                )
                combined_datasets = {
                    "train": train_valid["train"],
                    "validation": train_valid["test"]
                }
            else:
                # Use existing validation sets
                val_datasets = []
                for dataset in raw_datasets.values():
                    if "validation" in dataset:
                        val_datasets.append(dataset["validation"])
                
                combined_datasets = {
                    "train": combined_train,
                    "validation": concatenate_datasets(val_datasets) if val_datasets else None
                }
        else:
            raise ValueError("No training data found in any dataset")
    
    # Limit training samples for debugging or quick training
    if data_args.max_train_samples is not None and not data_args.streaming:
        combined_datasets["train"] = combined_datasets["train"].select(range(data_args.max_train_samples))
    
    # Limit validation samples
    if "validation" in combined_datasets and data_args.max_eval_samples is not None and not data_args.streaming:
        combined_datasets["validation"] = combined_datasets["validation"].select(range(data_args.max_eval_samples))
    
    # Apply tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=True)
    
    logger.info("Tokenizing datasets")
    
    if data_args.streaming:
        tokenized_datasets = {
            split: dataset.map(tokenize_function, batched=True)
            for split, dataset in combined_datasets.items()
        }
    else:
        tokenized_datasets = {
            split: dataset.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=["text"],
                desc=f"Tokenizing {split} set",
            )
            for split, dataset in combined_datasets.items()
        }
    
    # Group texts for more efficient training
    block_size = data_args.block_size
    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(f"The tokenizer selected has a max length of {tokenizer.model_max_length}, but we'll use block_size={block_size}")
    
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # We drop the small remainder, we could add padding if the model supported it
        total_length = (total_length // block_size) * block_size
        
        # Split by chunks of block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        
        # Create labels for causal language modeling
        result["labels"] = result["input_ids"].copy()
        return result
    
    logger.info(f"Grouping texts into blocks of size {block_size}")
    
    if data_args.streaming:
        lm_datasets = {
            split: dataset.map(group_texts, batched=True)
            for split, dataset in tokenized_datasets.items()
        }
    else:
        lm_datasets = {
            split: dataset.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc=f"Grouping texts for {split} set",
            )
            for split, dataset in tokenized_datasets.items()
        }
    
    return lm_datasets

def main():
    # Parse command line arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Initialize logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    
    # Set the verbosity to info of the Transformers logger
    transformers.utils.logging.set_verbosity_info()
    
    # Create output directory if needed
    if training_args.output_dir is not None:
        os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Set seed before initializing model
    set_seed(training_args.seed)
    
    # Load model or create a fresh one
    if model_args.model_path:
        logger.info(f"Loading model from {model_args.model_path}")
        model = MiniQwenForCausalLM.from_pretrained(
            model_args.model_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        logger.info("Creating a new MiniQwen model")
        model = create_fresh_model()
    
    # Load tokenizer
    tokenizer = create_tokenizer(model_args.tokenizer_path or model_args.model_path)
    
    # Load datasets
    logger.info(f"Loading datasets from config: {data_args.dataset_config}")
    datasets = load_and_process_dataset(data_args.dataset_config, tokenizer, data_args, training_args)
    
    if "validation" not in datasets:
        logger.info("No validation dataset provided, skipping validation")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We use causal language modeling instead of masked language modeling
    )
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"] if "train" in datasets else None,
        eval_dataset=datasets["validation"] if "validation" in datasets else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint := get_last_checkpoint(training_args.output_dir):
            checkpoint = last_checkpoint
            logger.info(f"Resuming training from checkpoint: {checkpoint}")
        
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        logger.info("Saving model checkpoint...")
        trainer.save_model()
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval and "validation" in datasets:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        logger.info(f"Evaluation results: {metrics}")
    
    # Save the tokenizer
    if training_args.output_dir is not None:
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Save dataset config info
        with open(os.path.join(training_args.output_dir, "dataset_info.json"), "w") as f:
            json.dump({
                "dataset_config": data_args.dataset_config,
                "pre_2020_sampling_rate": data_args.pre_2020_sampling_rate,
                "block_size": data_args.block_size,
                "dataset_sampling_rates": data_args.dataset_sampling_rates,
            }, f, indent=2)
    
    logger.info("Pretraining completed!")

if __name__ == "__main__":
    main()
