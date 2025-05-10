
"""Mini Qwen Tokenizer implementation."""

import os
from typing import Dict, List, Optional, Tuple, Union

import regex as re
from transformers import PreTrainedTokenizer
from transformers.utils import logging, PaddingStrategy

logger = logging.get_logger(__name__)

# 从BERT继承的词汇表常量
VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

# 特殊token的定义
SPECIAL_TOKENS_ATTRIBUTES = [
    "bos_token",
    "eos_token",
    "unk_token",
    "sep_token",
    "pad_token",
    "cls_token",
    "mask_token",
    "additional_special_tokens",
]


class MiniQwenTokenizer(PreTrainedTokenizer):
    """
    MiniQwen BPE tokenizer，基于BERT词表构建
    """
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

        # 加载词汇表
        if vocab_file is not None:
            self.vocab = self.load_vocab(vocab_file)
            self.ids_to_tokens = {ids: tok for tok, ids in self.vocab.items()}
        else:
            # 如果没有提供vocab_file，将尝试使用BERT词表
            from transformers import BertTokenizer
            bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.vocab = bert_tokenizer.vocab
            self.ids_to_tokens = bert_tokenizer.ids_to_tokens

        # 确保特殊token在词汇表中
        for special_token in [bos_token, eos_token, unk_token, sep_token, pad_token, cls_token, mask_token]:
            if special_token not in self.vocab:
                # 如果特殊token不在词汇表中，添加它
                self.add_tokens([special_token], special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """从文件加载词汇表"""
        vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        """基本的tokenization逻辑"""
        # 基于空格和标点符号分词
        basic_tokens = []
        for token in re.findall(r"\w+|[^\w\s]", text):
            # 当实现较复杂的BPE逻辑时，这里需要处理未知词
            if token not in self.vocab:
                basic_tokens.append(self.unk_token)
            else:
                basic_tokens.append(token)
        
        return basic_tokens

    def _convert_token_to_id(self, token: str) -> int:
        """将token转换为ID"""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        """将ID转换为token"""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """将tokens列表转换为字符串"""
        text = " ".join(tokens)
        # 处理空格
        text = text.replace(" ##", "")  # BERT风格的子词处理
        return text

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """插入特殊token"""
        bos_token_id = self.bos_token_id
        eos_token_id = self.eos_token_id
        
        output = [bos_token_id] + token_ids_0 + [eos_token_id]
        
        if token_ids_1 is not None:
            output += token_ids_1 + [eos_token_id]
            
        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """获取特殊token掩码"""
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        # 在序列开始和结束添加特殊token
        result = [1] + [0] * len(token_ids_0) + [1]
        
        if token_ids_1 is not None:
            result += [0] * len(token_ids_1) + [1]
            
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """创建token类型IDs"""
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """保存词汇表到文件"""
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
            
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, index in sorted(self.vocab.items(), key=lambda x: x[1]):
                writer.write(token + "\n")
                
        return (vocab_file,)
