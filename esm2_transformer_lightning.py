import os
import math
import torch
import numpy as np
import multiprocessing
import lightning as L
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from sklearn.metrics import mean_squared_error, r2_score


class KappaRAbDDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = os.listdir(self.data_folder)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_folder, file_name)
        
        data = torch.load(file_path)
        
        embedding = data['representations'][30]
        ddg = data['ddg']
        
        return embedding, ddg

def pad_collate_fn(batch):
    embeddings, ddgs = zip(*batch)
    padded_embeddings = pad_sequence(embeddings, batch_first=True, padding_value=0.0)
    padded_ddgs = torch.tensor(ddgs)
    
    return padded_embeddings, padded_ddgs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class TransformerRegressionModel(nn.Module):
    def __init__(self,input_size, hidden_size=256,num_transformer_layers=1, num_attention_heads=2):
        super(TransformerRegressionModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        #self.positional_encoding = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=256, dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.mlp1 = nn.Linear(hidden_size,64)
        self.mlp2 = nn.Linear(64,1)

    def forward(self, inputs):
        embedded_inputs = self.embedding(inputs)
        #positional_encoded_inputs = self.positional_encoding(embedded_inputs)
        positional_encoded_inputs = embedded_inputs
        transformer_output = self.transformer_encoder(positional_encoded_inputs)

        # Compute attention weights for each encoder layer
        attention_weights_list = []
        for layer in self.transformer_encoder.layers:
            attention_output, _ = layer.self_attn(embedded_inputs, transformer_output, transformer_output)
            attention_weights = F.softmax(attention_output)
            attention_weights_list.append(attention_weights)
        
        # Apply attention-based pooling for each encoder layer
        pooled_outputs = []
        for attention_weights in attention_weights_list:
            weighted_pooling = torch.einsum('blh,bld->bd', attention_weights, transformer_output)
            normalized_weights = attention_weights.sum(dim=1)
            pooled_output = weighted_pooling / normalized_weights
            pooled_outputs.append(pooled_output)
        
        # Concatenate pooled outputs from each encoder layer
        pooled_output = torch.cat(pooled_outputs, dim=-1)
        
        hidden = F.relu(self.mlp1(pooled_output))
        prediction = self.mlp2(hidden)
        return prediction

class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.model = model

        self.save_hyperparameters(ignore=["model"]) # If this is done, then we must specify the architecture when loading into

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        embeddings, ddgs = batch
        predictions = self(embeddings)
        loss = criterion(predictions,ddgs.view(-1,1))
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings, ddgs = batch
        predictions = self(embeddings)
        loss = criterion(predictions,ddgs.view(-1,1))
        self.log("test_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', True)
    data_folder = 'rabd/outputs_token/'
    dataset = KappaRAbDDataset(data_folder)
    
    train_dataset, test_dataset  = random_split(dataset,[int(0.85*len(dataset)), len(dataset) - int(0.85*len(dataset))])
    train_dataloader = DataLoader(train_dataset, batch_size=2000, shuffle=True, collate_fn=pad_collate_fn, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False, collate_fn=pad_collate_fn, num_workers=4)

    model = TransformerRegressionModel(input_size=640)
    criterion = nn.MSELoss()
    lightning_model = LightningModel(model=model, learning_rate=0.0001)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, mode="max", monitor="test_loss")

    trainer = L.Trainer(max_epochs=1000, accelerator="auto", devices="auto", logger=CSVLogger("logs/", name="model"), callbacks=[checkpoint_callback])
    trainer.fit(model=lightning_model,train_dataloaders=train_dataloader,val_dataloaders=test_dataloader)
