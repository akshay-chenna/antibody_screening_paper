import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device:{device}")

x_train = torch.load('x_train.pt')
y_train = torch.load('y_train.pt')
x_test = torch.load('x_test.pt')
y_test = torch.load('y_test.pt')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=250):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
    
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class TransformerRegressionModel(nn.Module):
    def __init__(self,input_size, hidden_size=64, num_transformer_layers=1, num_attention_heads=8):
        super().__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size).to(device)
        #self.positional_encoding = PositionalEncoding(input_size).to(device)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_attention_heads, dim_feedforward=60, dropout=0.3)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.mlp1 = nn.Linear(hidden_size,60)
        self.mlp2 = nn.Linear(60,1)

    def forward(self, inputs):
        embedded_inputs = self.embedding(inputs)
        positional_encoded_inputs = self.positional_encoding(embedded_inputs)
        #positional_encoded_inputs = self.positional_encoding(inputs)
        transformer_output = self.transformer_encoder(positional_encoded_inputs)

        # Compute attention weights for each encoder layer
        attention_weights_list = []
        for layer in self.transformer_encoder.layers:
            attention_output, _ = layer.self_attn(embedded_inputs, transformer_output, transformer_output)
            #attention_output, _ = layer.self_attn(inputs, transformer_output, transformer_output)
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
torch.manual_seed(88)

model = TransformerRegressionModel(input_size=640).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 1000
batch_size = 500  # size of each batch
batch_start = torch.arange(0, len(x_train), batch_size)
tbatch_start = torch.arange(0, len(x_test), batch_size)
train_loss_traj = []
test_loss_traj = []

min_test_loss = 36
for epoch in range(num_epochs):
    model = model.train()
    train_loss = 0
    n = 0
    for start in batch_start:
        batch_features = x_train[start:start+batch_size]
        batch_ddgs = y_train[start:start+batch_size]
        batch_features = batch_features.float().cuda()
        batch_ddgs = batch_ddgs.float().cuda()
        
        predictions = model(batch_features)
        batch_ddgs = batch_ddgs.view(-1, 1)
        loss = criterion(predictions, batch_ddgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n += 1
        
    if (epoch + 1) % 1 == 0:
        model = model.eval()
        test_loss = 0
        t = 0
        y_pred = []
        with torch.inference_mode():
            for tstart in tbatch_start:
                t_batch_features = x_test[tstart:tstart+batch_size]
                t_batch_ddgs = y_test[tstart:tstart+batch_size]
                t_batch_features = t_batch_features.float().cuda()
                t_batch_ddgs = t_batch_ddgs.float().cuda()
    
                t_predictions = model(t_batch_features)
                t_batch_ddgs = t_batch_ddgs.view(-1, 1)
                t_loss = criterion(t_predictions,t_batch_ddgs)
    
                test_loss += t_loss.item()
                t += 1
                y_pred.append(t_predictions.cpu().numpy())
    
            average_test_loss = test_loss / t
            average_train_loss = train_loss / n
            
            train_loss_traj.append(average_train_loss)
            test_loss_traj.append(average_test_loss)
            with open('model_8.txt','a') as file: print(f"Epoch: {epoch+1}, Train Loss: {average_train_loss:.4f}, Test Loss: {average_test_loss:.4f}", sep='\n', file=file)
            if average_train_loss < average_test_loss and average_test_loss < min_test_loss:
                torch.save(model,f"model_8_{average_test_loss:.4f}.pt")
                #min_test_loss = average_test_loss
