# main.py  

import torch  
from transformers import BertTokenizer  
from torch.optim import Adam  
from torch.nn import CrossEntropyLoss  
from tqdm import tqdm  
from config import (  
    NUM_EPOCHS,  
    BATCH_SIZE,  
    LEARNING_RATE,  
    DEVICE,  
    MODEL_DIR  
)  
from load import get_dataloader  
from model import PaperClassifier  
import os  

def train():  
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)  
    model = PaperClassifier().to(DEVICE)  
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)  
    criterion = CrossEntropyLoss()  

    train_loader = get_dataloader('train.jsonl', tokenizer, BATCH_SIZE)  
    valid_loader = get_dataloader('valid.jsonl', tokenizer, BATCH_SIZE, shuffle=False)  

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

if __name__ == '__main__':  
    train()