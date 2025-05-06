from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, concatenate_datasets
import torch

'''

增量预训练关键策略解析
​
​参数冻结​​
冻结全部参数后仅解冻最后2层，平衡新知识学习与旧知识保留
可通过调整解冻层数（如model.transformer.h[-4:]）控制训练参数量
​
​数据混合​​
新旧数据按3:7比例混合，缓解灾难性遗忘
使用concatenate_datasets实现动态数据采样

​​训练优化​​
余弦退火学习率：初始学习率2e-5，通过warmup_ratio实现平稳启动
混合精度训练（fp16=True）减少显存占用

'''

# 1. 加载基座模型与分词器（参考网页2、网页3）
model_name = "qwen/Qwen2-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

# 2. 冻结原始模型参数（增量预训练核心策略）
for param in model.parameters():
    param.requires_grad = False  # 冻结全部参数
# 仅解冻顶层模块（示例：最后2个Transformer块）
for layer in model.transformer.h[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

# 3. 准备混合数据集（参考网页6、网页7）
# 旧数据：通用领域（示例）
old_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# 新数据：目标领域（如医疗文本）
new_data = load_dataset("json", data_files="medical_data.jsonl")["train"]

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

# 混合比例：新数据30% + 旧数据70%（网页6提到的数据分布优化）
old_data = old_data.map(preprocess, batched=True).select_columns(["input_ids", "attention_mask"])
new_data = new_data.map(preprocess, batched=True).select_columns(["input_ids", "attention_mask"])
combined_data = concatenate_datasets([new_data, old_data]).shuffle(seed=42)

# 4. 配置训练参数（参考网页2、网页6的余弦退火策略）
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=2e-5,
    warmup_ratio=0.06,  # 预热阶段占6%（网页6的YARN策略相似）
    lr_scheduler_type="cosine",  # 余弦退火调度
    gradient_accumulation_steps=8,
    fp16=True,  # 混合精度训练
    save_strategy="epoch",
    logging_steps=100,
)

# 5. 创建训练器并启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=combined_data,
    data_collator=lambda data: {
        "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]),
        "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]),
        "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data]),  # 自回归预训练
    }
)
trainer.train()

# 6. 保存增量后的模型
model.save_pretrained("./qwen2-incremental")
tokenizer.save_pretrained("./qwen2-incremental")



class PretrainDataset():
    '''
    接收经过分词的数据集
    '''
    def __init__(self, dataset, max_seq_length:int = 512):
        self.dataset = dataset
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.dataset)

    
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        input_ids = sample["input_ids"]
        attention_mask = sample["attention_mask"]
        labels = input_ids.clone() # 自回归预训练, 因此，标签就是 input_ids 左移一位
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }



class ContinuePreTrainer(Trainer):
    
    
    def __init__(self, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer



class ContinuePreTrainerWrapper():
    def __init__(
        self,
        model_name_or_path:str = None,
        new_data_path:str = None,
        old_data_path:str = None,
        output_dir:str = None,
        ):
        self.model_name_or_path = model_name_or_path
        self.new_data_path = new_data_path
        self.old_data_path = old_data_path
        self.output_dir = output_dir
        
        
        self.training_args = TrainingArguments(
            output_dir="./output",
            per_device_train_batch_size=4,
            num_train_epochs=3,
            learning_rate=2e-5,
            warmup_ratio=0.06,  # 预热阶段占6%（网页6的YARN策略相似）
            lr_scheduler_type="cosine",  # 余弦退火调度
            gradient_accumulation_steps=8,
            fp16=True,  # 混合精度训练
            save_strategy="epoch",
            logging_steps=100,
        )
        
        
        self.trainer = trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=combined_data,
            data_collator=lambda data: {
                "input_ids": torch.stack([torch.tensor(d["input_ids"]) for d in data]),
                "attention_mask": torch.stack([torch.tensor(d["attention_mask"]) for d in data]),
                "labels": torch.stack([torch.tensor(d["input_ids"]) for d in data]),  # 自回归预训练
            }
        )
        
        
        
    def _initialize_model_and_tokenizer(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path)
    
    def train(self):
        self.trainer.train()
        
        
    def save(self, save_path:str = None):
        
        if save_path is None:
            save_path = self.training_args.output_dir
        self.trainer.save_model(save_path)
        
        
    
if __name__ == "__main__":
    training_wrapper = ContinuePreTrainerWrapper()
    training_wrapper.train()