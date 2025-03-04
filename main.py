# main.py  

import torch  
import torch.nn as nn
from transformers import BertTokenizer  
from torch.optim import Adam  
from torch.nn import CrossEntropyLoss  
from tqdm import tqdm  
from src.configs.config import (  
    NUM_EPOCHS,  
    BATCH_SIZE,  
    LEARNING_RATE,  
    DEVICE,  
    MODEL_DIR,
    LLAMA_MODEL_PATH,
    LLAMA_ADAPTER_PATH,
    LLAMA_TRAINED_PATH,
    BERT_MODEL_PATH,
    LLAMA_TOKENIZER_PATH,
)  

from src.models.model import PaperClassifier  

from transformers import AutoTokenizer, get_linear_schedule_with_warmup  
from src.models.model import IntentClassifier  
from src.data.data_preprocess import load_data  
from src.evaluation.evaluate import Evaluator 

import os  

def train():  
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)  
    model = PaperClassifier().to(DEVICE)  
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)  
    criterion = CrossEntropyLoss()  

    train_loader = load_data('train.jsonl', tokenizer, BATCH_SIZE)  
    valid_loader = load_data('valid.jsonl', tokenizer, BATCH_SIZE, shuffle=False)  

    for epoch in range(NUM_EPOCHS):  
        model.train()  
        total_loss = 0  
        for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{NUM_EPOCHS}'):  
            input_ids = batch['input_ids'].to(DEVICE)  
            attention_mask = batch['attention_mask'].to(DEVICE)  
            labels = batch['label'].to(DEVICE)  

            optimizer.zero_grad()  
            outputs = model(input_ids, attention_mask)  
            loss = criterion(outputs, labels)  
            loss.backward()  
            optimizer.step()  

            total_loss += loss.item()  

        avg_loss = total_loss / len(train_loader)  
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}')  

        # 可以添加验证逻辑  

    # 保存模型  
    os.makedirs(MODEL_DIR, exist_ok=True)  
    model_save_path = os.path.join(MODEL_DIR, 'paper_classifier.pt')  
    torch.save(model.state_dict(), model_save_path)  
    print(f'Model saved to {model_save_path}')  
    
    

def train_llama():
    '''
        训练基于llama3的意图预测模型
    '''
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)  
    train_loader, val_loader = load_data(tokenizer)  
    
    model = IntentClassifier().to(DEVICE)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)  
    total_steps = len(train_loader) * NUM_EPOCHS  
    scheduler = get_linear_schedule_with_warmup(  
        optimizer,  
        num_warmup_steps=0,  
        num_training_steps=total_steps  
    )  
    criterion = nn.CrossEntropyLoss()  

    evaluator = Evaluator(model, tokenizer)  

    for epoch in range(NUM_EPOCHS):  
        model.train()  
        total_loss = 0  
        for batch in train_loader:  
            optimizer.zero_grad()  

            input_ids = batch['input_ids'].to(DEVICE)  
            attention_mask = batch['attention_mask'].to(DEVICE)  
            labels = batch['labels'].to(DEVICE)  

            outputs = model(  
                input_ids=input_ids,  
                attention_mask=attention_mask,  
                labels=labels  
            )  
            loss = outputs.loss  
            loss.backward()  
            optimizer.step()  
            scheduler.step()  

            total_loss += loss.item()  

        avg_train_loss = total_loss / len(train_loader)  
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Training Loss: {avg_train_loss}")  

        # 验证模型性能  
        val_accuracy = evaluator.evaluate(val_loader)  
        print(f"Validation Accuracy: {val_accuracy}")  

    # 保存模型  
    model_save_path = LLAMA_TRAINED_PATH
    model.model.save_pretrained(model_save_path)  
    tokenizer.save_pretrained(LLAMA_TOKENIZER_PATH)  

if __name__ == '__main__':  
    train()