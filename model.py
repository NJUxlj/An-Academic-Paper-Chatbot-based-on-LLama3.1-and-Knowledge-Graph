# model.py  

import torch  
import torch.nn as nn  
from transformers import BertModel  
from config import NUM_CLASSES, BERT_MODEL_PATH  

class PaperClassifier(nn.Module):  
    """  
    论文分类模型，使用预训练的BERT模型作为编码器。  
    """  
    def __init__(self):  
        super(PaperClassifier, self).__init__()  
        self.bert = BertModel.from_pretrained(BERT_MODEL_PATH)  
        self.dropout = nn.Dropout(0.1)  
        self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_CLASSES)  

    def forward(self, input_ids, attention_mask):  
        outputs = self.bert(  
            input_ids=input_ids,  
            attention_mask=attention_mask  
        )  
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS]向量  
        pooled_output = self.dropout(pooled_output)  
        logits = self.classifier(pooled_output)  
        return logits  

class EntityRelationExtractor(nn.Module):  
    """  
    实体和关系抽取模型，使用BERT+BiLSTM+CRF架构。  
    """  
    def __init__(self, tag_set_size):  
        super(EntityRelationExtractor, self).__init__()  
        self.bert = BertModel.from_pretrained(BERT_MODEL_PATH)  
        self.lstm = nn.LSTM(  
            input_size=self.bert.config.hidden_size,  
            hidden_size=256,  
            num_layers=1,  
            bidirectional=True,  
            batch_first=True  
        )  
        self.fc = nn.Linear(256 * 2, tag_set_size)  
        self.crf = CRF(tag_set_size, batch_first=True)  

    def forward(self, input_ids, attention_mask, tags=None):  
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  
        sequence_output = bert_outputs.last_hidden_state  
        lstm_output, _ = self.lstm(sequence_output)  
        emissions = self.fc(lstm_output)  
        if tags is not None:  
            loss = -self.crf(emissions, tags, mask=attention_mask.byte())  
            return loss  
        else:  
            prediction = self.crf.decode(emissions, mask=attention_mask.byte())  
            return prediction