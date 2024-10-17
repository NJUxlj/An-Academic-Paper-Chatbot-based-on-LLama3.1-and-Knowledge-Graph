# load.py  

import os  
import json  
from torch.utils.data import Dataset, DataLoader  
from transformers import BertTokenizer  
from config import DATA_DIR, MAX_SEQ_LENGTH  

class PaperDataset(Dataset):  
    """  
    自定义数据集类，用于加载论文数据。  
    """  
    def __init__(self, file_path, tokenizer, max_seq_length=MAX_SEQ_LENGTH):  
        self.samples = []  
        self.tokenizer = tokenizer  
        self.max_seq_length = max_seq_length  

        full_path = os.path.join(DATA_DIR, file_path)  
        with open(full_path, 'r', encoding='utf-8') as f:  
            for line in f:  
                sample = json.loads(line.strip())  
                self.samples.append(sample)  

    def __len__(self):  
        return len(self.samples)  

    def __getitem__(self, idx):  
        sample = self.samples[idx]  
        text = sample['text']  
        label = sample['label']  

        inputs = self.tokenizer.encode_plus(  
            text,  
            None,  
            add_special_tokens=True,  
            max_length=self.max_seq_length,  
            padding='max_length',  
            truncation=True,  
            return_tensors='pt'  
        )  

        return {  
            'input_ids': inputs['input_ids'].squeeze(0),  
            'attention_mask': inputs['attention_mask'].squeeze(0),  
            'label': label  
        }  

def get_dataloader(file_path, tokenizer, batch_size, shuffle=True):  
    dataset = PaperDataset(file_path, tokenizer)  
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  
    return dataloader