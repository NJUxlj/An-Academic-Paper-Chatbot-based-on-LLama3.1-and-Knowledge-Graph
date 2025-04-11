

# src/models/entity_extractor/bert_bilstm_crf.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertTokenizer
from torchcrf import CRF
import logging
from typing import List, Dict, Tuple, Set, Optional, Any

logger = logging.getLogger(__name__)

class EntityTripleExtractor:
    """
    使用Bert+BiLSTM+CRF模型进行实体关系抽取
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化抽取器
        
        Args:
            model_path: BERT模型路径
            device: 使用的设备 (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BERT_BiLSTM_CRF.from_pretrained(model_path)
        
        # 实体和关系标签
        self.entity_labels = ["O", "B-MODEL", "I-MODEL", "B-METRIC", "I-METRIC", 
                             "B-DATASET", "I-DATASET", "B-METHOD", "I-METHOD", 
                             "B-TASK", "I-TASK", "B-FRAMEWORK", "I-FRAMEWORK"]
        
        self.relation_types = ["uses", "evaluates_on", "achieves", "improves", 
                               "applies_to", "proposes", "part_of", "has_attribute"]
        
        # 迁移模型到指定设备
        self.model.to(self.device)
        self.model.eval()
    
    def predict_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中预测实体
        
        Args:
            text: 输入文本
            
        Returns:
            预测的实体列表，包含类型和位置
        """
        # 分词处理
        tokens = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # 模型输入准备
        inputs = {
            'input_ids': torch.tensor([input_ids], device=self.device),
            'attention_mask': torch.tensor([[1] * len(input_ids)], device=self.device)
        }
        
        # 执行预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs[0]
        
        # 解析预测结果
        entities = []
        current_entity = None
        
        for i, pred in enumerate(predictions[0]):
            tag = self.entity_labels[pred]
            
            if tag.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = tag[2:]  # 去掉'B-'前缀
                current_entity = {
                    'type': entity_type,
                    'start': i,
                    'end': i,
                    'text': tokens[i]
                }
            
            elif tag.startswith('I-') and current_entity:
                entity_type = tag[2:]  # 去掉'I-'前缀
                
                # 确保当前词属于同一实体类型
                if entity_type == current_entity['type']:
                    current_entity['end'] = i
                    current_entity['text'] += ' ' + tokens[i]
            
            elif tag == 'O':
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 添加最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def extract_triples(self, text: str) -> List[Dict[str, str]]:
        """
        从文本中提取三元组
        
        Args:
            text: 输入文本
            
        Returns:
            三元组列表 (头实体, 关系, 尾实体)
        """
        # 1. 先提取所有实体
        entities = self.predict_entities(text)
        
        # 2. 按距离和语义关联构建三元组
        triples = []
        
        # 简单启发式方法：相邻实体之间可能有关系
        for i in range(len(entities) - 1):
            head_entity = entities[i]
            tail_entity = entities[i + 1]
            
            # 提取两个实体之间的文本
            middle_text = text[head_entity['end']:tail_entity['start']]
            
            # 根据实体类型和中间文本推断关系
            relation = self._infer_relation(head_entity, tail_entity, middle_text)
            
            if relation:
                triple = {
                    'head': head_entity['text'],
                    'head_type': head_entity['type'],
                    'relation': relation,
                    'tail': tail_entity['text'],
                    'tail_type': tail_entity['type']
                }
                triples.append(triple)
        
        return triples
    
    def _infer_relation(self, head_entity: Dict, tail_entity: Dict, middle_text: str) -> Optional[str]:
        """
        根据实体类型和中间文本推断关系
        
        Args:
            head_entity: 头实体
            tail_entity: 尾实体
            middle_text: 两个实体之间的文本
            
        Returns:
            推断出的关系类型，如果无法推断则返回None
        """
        # 关系映射规则
        relation_rules = {
            ('MODEL', 'DATASET'): 'evaluates_on' if 'evaluate' in middle_text or 'test' in middle_text else 'uses',
            ('MODEL', 'METRIC'): 'achieves' if 'achieve' in middle_text or 'reach' in middle_text else None,
            ('METHOD', 'TASK'): 'applies_to' if 'apply' in middle_text or 'used for' in middle_text else None,
            ('MODEL', 'METHOD'): 'uses' if 'use' in middle_text or 'employ' in middle_text else None,
            ('FRAMEWORK', 'MODEL'): 'part_of' if 'contain' in middle_text or 'include' in middle_text else None,
        }
        
        # 检查实体类型对是否存在于规则中
        pair = (head_entity['type'], tail_entity['type'])
        if pair in relation_rules:
            return relation_rules[pair]
        
        # 处理属性关系
        if 'has' in middle_text or 'with' in middle_text:
            return 'has_attribute'
        
        # 基于常见动词的关系提取
        if 'propose' in middle_text or 'introduce' in middle_text:
            return 'proposes'
        elif 'improve' in middle_text or 'enhance' in middle_text:
            return 'improves'
        
        return None


class BERT_BiLSTM_CRF(nn.Module):
    """
    BERT+BiLSTM+CRF模型用于命名实体识别
    """
    
    def __init__(self, bert_model: str, num_tags: int, lstm_hidden_dim: int = 256, dropout: float = 0.1):
        """
        初始化模型
        
        Args:
            bert_model: BERT模型名称或路径
            num_tags: 标签数量
            lstm_hidden_dim: LSTM隐藏层维度
            dropout: Dropout概率
        """
        super(BERT_BiLSTM_CRF, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model)
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(lstm_hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        前向传播
        
        Args:
            input_ids: 输入序列
            attention_mask: 注意力掩码
            labels: 标签序列
            
        Returns:
            根据是否提供标签返回不同的输出
        """
        # BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        
        sequence_output = outputs[0]
        
        # BiLSTM
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output)
        
        # 线性层
        emissions = self.hidden2tag(lstm_output)
        
        # 如果提供了标签，计算损失
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss
        else:
            # CRF解码
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions
    
    @classmethod
    def from_pretrained(cls, bert_model_path: str, num_tags: int = 13):
        """
        从预训练BERT模型创建实例
        
        Args:
            bert_model_path: BERT模型路径
            num_tags: 标签数量
            
        Returns:
            模型实例
        """
        model = cls(bert_model_path, num_tags)
        return model

