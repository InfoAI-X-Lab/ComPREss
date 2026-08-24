import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinClassifier(nn.Module):
    def __init__(self, esm_model, config, pretrained_path=None, is_pretrain=True):
        super().__init__()
        self.esm = esm_model
        self.is_pretrain = is_pretrain
        for param in self.esm.parameters():
            param.requires_grad = False
        hidden_size = self.esm.config.hidden_size
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=4*hidden_size,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, config['hidden_dim2']),
            nn.LayerNorm(config['hidden_dim2']),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(config['hidden_dim2'], config['projection_dim'])
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, config['hidden_dim1']),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(config['hidden_dim1'], config['hidden_dim2']),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            nn.Linear(config['hidden_dim2'], 6)
        )
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            pretrained_dict = torch.load(pretrained_path, map_location='cpu')
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                             if k in model_dict and ('transformer' in k or 'mlm_head' in k)}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Loaded {len(pretrained_dict)} pretrained parameters")
        self._reinitialize_heads()
    
    def _reinitialize_heads(self):
        print("Reinitializing projection_head and classifier weights with Kaiming initialization for ReLU activation...")
        # 重新初始化projection_head - 使用Kaiming初始化
        for module in self.projection_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    def forward(self, input_ids, attention_mask, labels=None, mlm_labels=None, return_embeddings=False, return_classification=False):
        if input_ids.dim() == 3:
            batch_size, num_windows, window_length = input_ids.size()
            input_ids = input_ids.view(batch_size * num_windows, window_length)
            attention_mask = attention_mask.view(batch_size * num_windows, window_length)
        else:
            batch_size = input_ids.size(0)
            num_windows = 1
        with torch.no_grad():
            outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # [batch_size * num_windows, seq_length, hidden_size]
        transformer_output = self.transformer(embeddings)
        pooled_output = torch.mean(transformer_output, dim=1)  # [batch_size * num_windows, hidden_size]
        if num_windows > 1:
            pooled_output = pooled_output.view(batch_size, num_windows, -1)
            pooled_output = torch.mean(pooled_output, dim=1)  # [batch_size, hidden_size]
        if return_embeddings:
            return pooled_output
        if self.is_pretrain:
            mlm_output = self.mlm_head(transformer_output)
            if mlm_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                mlm_loss = loss_fct(mlm_output.view(-1, self.esm.config.vocab_size), mlm_labels.view(-1))
                return mlm_loss
            return mlm_output
        else:
            projections = self.projection_head(pooled_output)
            projections = F.normalize(projections, p=2, dim=1, eps=1e-8)
            if return_classification:
                classification_output = self.classifier(pooled_output)
                return projections, classification_output
            return projections
