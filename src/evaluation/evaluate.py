# evaluate.py  

import torch  
import numpy as np
from sklearn.metrics import classification_report  
from transformers import BertTokenizer  
from src.data.data_preprocess import get_dataloader  
from src.models.model import PaperClassifier  
from src.configs.config import (  
    BATCH_SIZE,  
    DEVICE,  
    MODEL_DIR,  
    BERT_MODEL_PATH,  
    CATEGORY_NAMES      
)  
import os  
import tqdm

class Evaluator:  
    """  
    模型评估类  
    """  
    def __init__(self, model_path):  
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)  
        self.model = PaperClassifier().to(DEVICE)  
        self.model.load_state_dict(torch.load(model_path))  

    def eval(self, data_file):  
        dataloader = get_dataloader(data_file, self.tokenizer, BATCH_SIZE, shuffle=False)  
        self.model.eval()  
        all_labels = []  
        all_preds = []  

        with torch.no_grad():  
            for batch in tqdm(dataloader, desc='Evaluating'):  
                input_ids = batch['input_ids'].to(DEVICE)  
                attention_mask = batch['attention_mask'].to(DEVICE)  
                labels = batch['label'].to(DEVICE)  

                outputs = self.model(input_ids, attention_mask)  
                preds = torch.argmax(outputs, dim=1)  

                all_labels.extend(labels.cpu().numpy())  
                all_preds.extend(preds.cpu().numpy())  

        self.all_labels = all_labels  
        self.all_preds = all_preds  

    def write_stats(self, file_path):  
        report = classification_report(self.all_labels, self.all_preds, target_names=CATEGORY_NAMES)  
        with open(file_path, 'w', encoding='utf-8') as f:  
            f.write(report)  

    def show_stats(self):  
        report = classification_report(self.all_labels, self.all_preds, target_names=CATEGORY_NAMES)  
        print(report)  

    def decode(self, label):  
        return CATEGORY_NAMES[label]  

    def write_stats_to_csv(self, file_path):  
        import pandas as pd  
        report = classification_report(self.all_labels, self.all_preds, target_names=CATEGORY_NAMES, output_dict=True)  
        df = pd.DataFrame(report).transpose()  
        df.to_csv(file_path, index=True)  
        
        
        
        

class LLaMAEvaluator:
    def __init__(self, model, tokenizer):  
        self.model = model  
        self.tokenizer = tokenizer  
    
    def eval(self, data_loader):  
        self.model.eval()  
        predictions = []  
        true_labels = []  

        with torch.no_grad():  
            for batch in data_loader:  
                input_ids = batch['input_ids'].to(DEVICE)  
                attention_mask = batch['attention_mask'].to(DEVICE)  
                labels = batch['labels'].to(DEVICE)  

                outputs = self.model(  
                    input_ids=input_ids,  
                    attention_mask=attention_mask  
                )  
                logits = outputs.logits  
                preds = torch.argmax(logits, dim=1)  

                predictions.extend(preds.cpu().numpy())  
                true_labels.extend(labels.cpu().numpy())  
        
        return true_labels, predictions  

    def write_stats(self, true_labels, predictions, file_path):  
        report = classification_report(  
            true_labels, predictions, target_names=INTENT_LABELS  
        )  
        with open(file_path, 'w') as f:  
            f.write(report)  

    def show_stats(self, true_labels, predictions):  
        report = classification_report(  
            true_labels, predictions, target_names=INTENT_LABELS  
        )  
        print(report)  

    def decode(self, preds):  
        idx_to_label = {idx: label for idx, label in enumerate(INTENT_LABELS)}  
        decoded_preds = [idx_to_label[pred] for pred in preds]  
        return decoded_preds  

    def write_stats_to_csv(self, true_labels, predictions, file_path):  
        import pandas as pd  
        decoded_preds = self.decode(predictions)  
        decoded_true = self.decode(true_labels)  
        df = pd.DataFrame({  
            'True Label': decoded_true,  
            'Predicted Label': decoded_preds  
        })  
        df.to_csv(file_path, index=False)  

    def evaluate(self, data_loader):  
        true_labels, predictions = self.eval(data_loader)  
        self.show_stats(true_labels, predictions)  
        accuracy = (np.array(true_labels) == np.array(predictions)).mean()  
        return accuracy  
    
    


if __name__ == '__main__':  
    model_path = os.path.join(MODEL_DIR, 'paper_classifier.pt')  
    evaluator = Evaluator(model_path)  
    evaluator.eval('test.jsonl')  
    evaluator.show_stats()  
    evaluator.write_stats('evaluation_report.txt')  
    evaluator.write_stats_to_csv('evaluation_report.csv')