import torch
import torch.nn as nn


class ProteinPretrainModel(nn.Module):
    def __init__(self, esm_model, config=None, is_pretrain=True):
        super().__init__()
        self.esm = esm_model
        self.is_pretrain = is_pretrain
        if config is None:
            config = {'hidden_dim1': 128, 'hidden_dim2': 512, 'dropout_rate': 0.1}
        hidden_dim1 = config.get('hidden_dim1', 128)
        hidden_dim2 = config.get('hidden_dim2', 512)
        dropout_rate = config.get('dropout_rate', 0.1)
        for param in self.esm.parameters():
            param.requires_grad = False
        trainable_params = [n for n, p in self.named_parameters() if p.requires_grad]
        print("Trainable parameters:", trainable_params)
        hidden_size = self.esm.config.hidden_size
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=4 * hidden_size,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.esm.config.vocab_size)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 6)
        )

    def forward(self, input_ids, attention_mask, labels=None, mlm_labels=None):
        if input_ids.dim() == 3:  # [batch_size, num_windows, window_length]
            batch_size, num_windows, window_length = input_ids.size()
            input_ids = input_ids.view(batch_size * num_windows, window_length)
            attention_mask = attention_mask.view(batch_size * num_windows, window_length)
        with torch.no_grad():
            outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        if not self.is_pretrain:
            embeddings = embeddings.view(batch_size, num_windows * embeddings.size(1), -1)

        transformer_input = embeddings.transpose(0, 1)
        transformer_output = self.transformer(transformer_input)
        transformer_output = transformer_output.transpose(0, 1)

        if self.is_pretrain:
            mlm_output = self.mlm_head(transformer_output)
            if mlm_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                mlm_loss = loss_fct(mlm_output.view(-1, self.esm.config.vocab_size), mlm_labels.view(-1))
                return mlm_loss
            return mlm_output
        else:
            pooled_output = torch.mean(transformer_output, dim=1)  # [batch_size, hidden_size]
            logits = self.classifier(pooled_output)
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss
            return logits
