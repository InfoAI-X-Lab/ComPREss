"""Training loops for one fold and one learner."""

import gc
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import BalancedBatchSampler, ContrastiveDataset, contrastive_collate_fn
from losses import CenterLoss, EnhancedClassAwareSupConLoss
from models.classifier_model import ProteinClassifier
from utils import set_seed


def train_fold(fold_idx, train_class_sequences, test_class_sequences, tokenizer, esm_model, device, learner_config, learner_seed, learner_name):
    print(f"\n=== Training Fold {fold_idx + 1} / Learner: {learner_name} (Seed: {learner_seed}) ===")
    set_seed(learner_seed)
    train_dataset = ContrastiveDataset(train_class_sequences, tokenizer,
                                     learner_config['max_length'], learner_config['overlap'])
    sampler = BalancedBatchSampler(train_dataset, learner_config['batch_size'], drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler,
                            collate_fn=contrastive_collate_fn)
    model = ProteinClassifier(
        esm_model,
        config=learner_config,
        pretrained_path=learner_config.get('pretrained_transformer_path'),
        is_pretrain=False
    ).to(device)

    print("Stage 1: Contrastive learning...")
    for name, param in model.named_parameters():
        if 'esm' in name:
            param.requires_grad = False
        elif 'transformer' in name or 'projection_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    stage1_lr = learner_config['stage1_learning_rate']
    weight_decay = learner_config['weight_decay']
    temperature = learner_config['temperature']
    center_loss_start_epoch = learner_config['center_loss_weight_start_epoch']
    center_loss_max_weight = learner_config['center_loss_weight_max']
    optimizer_stage1 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                       lr=stage1_lr,
                                       weight_decay=weight_decay)
    criterion_contrastive = EnhancedClassAwareSupConLoss(temperature=temperature)
    center_loss = CenterLoss(num_classes=6, feature_dim=learner_config['projection_dim'], alpha=0.01, exclude_class=2).to(device)
    for epoch in range(learner_config['num_epochs']):
        model.train()
        total_contrastive_loss = 0
        total_center_loss = 0
        total_combined_loss = 0
        if epoch >= center_loss_start_epoch:
            center_loss_weight = center_loss_max_weight
        else:
            center_loss_weight = 0.0
        for batch in train_loader:
            optimizer_stage1.zero_grad()
            anchor_input_ids = batch['anchor']['input_ids'].to(device)
            anchor_attention_mask = batch['anchor']['attention_mask'].to(device)
            positive_input_ids = batch['positive']['input_ids'].to(device)
            positive_attention_mask = batch['positive']['attention_mask'].to(device)
            negative_input_ids = batch['negatives']['input_ids'].to(device)
            negative_attention_mask = batch['negatives']['attention_mask'].to(device)
            labels = torch.tensor(batch['labels']).to(device)
            anchor_proj = model(anchor_input_ids, anchor_attention_mask)
            positive_proj = model(positive_input_ids, positive_attention_mask)
            negative_proj = model(negative_input_ids, negative_attention_mask)
            features = torch.stack([anchor_proj, positive_proj], dim=1)
            contrastive_loss = criterion_contrastive(features, labels)
            if center_loss_weight > 0:
                center_loss_value = center_loss(anchor_proj, labels)
                total_loss = contrastive_loss + center_loss_weight * center_loss_value
            else:
                total_loss = contrastive_loss
                center_loss_value = torch.tensor(0.0, device=device)
            total_loss.backward()
            if center_loss_weight > 0:
                with torch.no_grad():
                    center_loss.update_centers(anchor_proj, labels)
            optimizer_stage1.step()
            total_contrastive_loss += contrastive_loss.item()
            total_center_loss += center_loss_value.item()
            total_combined_loss += total_loss.item()
        avg_contrastive_loss = total_contrastive_loss / len(train_loader)
        avg_center_loss = total_center_loss / len(train_loader)
        avg_combined_loss = total_combined_loss / len(train_loader)

        if epoch < center_loss_start_epoch:
            print(f'Stage 1 Epoch {epoch + 1} (Contrastive Only), Contrastive Loss: {avg_contrastive_loss:.4f}, Temperature: {temperature:.4f}')
        else:
            print(f'Stage 1 Epoch {epoch + 1} (Contrastive + Center), Contrastive Loss: {avg_contrastive_loss:.4f}, Center Loss: {avg_center_loss:.4f}, Combined Loss: {avg_combined_loss:.4f}, Temperature: {temperature:.4f}')
        if (epoch + 1) % 5 == 0 and epoch >= center_loss_start_epoch:
            with torch.no_grad():
                center_distances = torch.norm(center_loss.centers.unsqueeze(0) - center_loss.centers.unsqueeze(1), dim=2)
                print(f'class center distance matrix (Epoch {epoch + 1}):')
                for i in range(6):
                    if i != 2:
                        distances = [f'{center_distances[i][j].item():.4f}' if j != 2 else 'N/A' for j in range(6)]
                        print(f'  Class {i}: {distances}')

    print("Stage 2: Downstream classification - Training classifier...")
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    stage2_learning_rate = learner_config['stage2_learning_rate']
    stage2_dropout_rate = learner_config['stage2_dropout_rate']
    weight_decay = learner_config['weight_decay']
    if hasattr(model.classifier[2], 'p'):
        model.classifier[2].p = stage2_dropout_rate
    if hasattr(model.classifier[5], 'p'):
        model.classifier[5].p = stage2_dropout_rate
    print(f"Set classifier dropout rate to: {stage2_dropout_rate}")
    optimizer_stage2 = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 
                                       lr=stage2_learning_rate,
                                       weight_decay=weight_decay)
    class_weights = torch.tensor([1.0, 1.0, 3.5, 2.5, 3.5, 3.5]).to(device)
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    best_val_loss = float('inf')
    patience = learner_config.get('patience', 4)
    min_delta = 0.0005
    no_improve_epochs = 0
    output_dir = learner_config.get('output_dir', '.')
    os.makedirs(output_dir, exist_ok=True)
    best_model_path = os.path.join(output_dir, f'best_model_fold_{fold_idx + 1}_{learner_name}.pth')
    for epoch in range(learner_config['num_epochs_stage2']):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer_stage2.zero_grad()
            anchor_input_ids = batch['anchor']['input_ids'].to(device)
            anchor_attention_mask = batch['anchor']['attention_mask'].to(device)
            labels = torch.tensor(batch['labels']).to(device)
            _, classification_output = model(anchor_input_ids, anchor_attention_mask, return_classification=True)
            loss = criterion_cls(classification_output, labels)
            loss.backward()
            optimizer_stage2.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f'Stage 2 Epoch {epoch + 1}, Classification Loss: {avg_loss:.4f}')
        if avg_loss < best_val_loss - min_delta:
            best_val_loss = avg_loss
            no_improve_epochs = 0
            if not learner_config.get('is_tuning', False):
                torch.save(model.state_dict(), best_model_path)
                print(f'Best model for {learner_name} saved with loss: {best_val_loss:.4f} to {best_model_path}')
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f'Early stopping for {learner_name} triggered at epoch {epoch + 1}')
            break
    if not learner_config.get('is_tuning', False) and os.path.exists(best_model_path):
        print(f"Loading best saved weights for {learner_name} from {best_model_path} for evaluation")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    del train_loader, train_dataset, optimizer_stage1, optimizer_stage2, criterion_contrastive, center_loss
    torch.cuda.empty_cache()
    gc.collect()
    return model
