"""Evaluation, metrics, and prediction export."""

import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import TestDataset, test_collate_fn

def _create_test_loader(test_class_sequences, tokenizer, max_length=1024, overlap=100):
    test_dataset = TestDataset(test_class_sequences, tokenizer, max_length, overlap)
def _compute_metrics(labels, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None, labels=range(6))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(labels, predictions, average='micro')
    cm = confusion_matrix(labels, predictions, labels=range(6))
    return {
        'accuracy': accuracy_score(labels, predictions),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        'confusion_matrix': cm,
        'predictions': predictions,
        'labels': labels
    }
def _print_metrics(metrics, title):
    print(f"\n=== {title} Results ===")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Average - Precision: {metrics['macro_precision']:.4f}, "
        f"Recall: {metrics['macro_recall']:.4f}, F1: {metrics['macro_f1']:.4f}")
    print(f"Micro Average - Precision: {metrics['micro_precision']:.4f}, "
        f"Recall: {metrics['micro_recall']:.4f}, F1: {metrics['micro_f1']:.4f}")
    print("\nPer-class metrics:")
    for i in range(6):
        print(f"Class {i}: Precision: {metrics['per_class_precision'][i]:.4f}, "
            f"Recall: {metrics['per_class_recall'][i]:.4f}, "
            f"F1: {metrics['per_class_f1'][i]:.4f}, "
            f"Support: {metrics['per_class_support'][i]}")
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(classification_report(
        metrics['labels'],
        metrics['predictions'],
        target_names=[f'Class_{i}' for i in range(6)]))

def evaluate_model(model, test_class_sequences, tokenizer, device, fold_idx):
    print(f"\n=== Evaluating Fold {fold_idx + 1} ===")
    test_loader = _create_test_loader(test_class_sequences, tokenizer)
    model.eval()
    all_predictions, all_labels = [], []
    print("Predicting test samples...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            _, logits = model(input_ids, attention_mask, return_classification=True)
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels)
    metrics = _compute_metrics(all_labels, all_predictions)
    _print_metrics(metrics, f"Fold {fold_idx + 1}")
    return metrics


def evaluate_ensemble(models, test_class_sequences, tokenizer, device, fold_idx, model_names=None):
    print(f"\n=== Evaluating Ensemble for Fold {fold_idx + 1} ===")
    test_loader = _create_test_loader(test_class_sequences, tokenizer)
    if model_names is None:
        model_names = [f'learner_{i + 1}' for i in range(len(models))]
    all_predictions, all_labels = [], []
    learner_predictions = {name: [] for name in model_names}
    print("Ensemble predicting test samples (soft-vote)...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            probs_sum = None
            for model, name in zip(models, model_names):
                model.eval()
                _, logits = model(input_ids, attention_mask, return_classification=True)
                probs = torch.softmax(logits, dim=1)
                learner_predictions[name].extend(torch.argmax(probs, dim=1).cpu().numpy())
                probs_sum = probs if probs_sum is None else probs_sum + probs
            predictions = torch.argmax(probs_sum / len(models), dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels)
    ensemble_metrics = _compute_metrics(all_labels, all_predictions)
    _print_metrics(ensemble_metrics, f"Fold {fold_idx + 1} Ensemble")
    learner_metrics = {}
    for name, preds in learner_predictions.items():
        metrics = _compute_metrics(all_labels, preds)
        learner_metrics[name] = metrics
        _print_metrics(metrics, f"Fold {fold_idx + 1} {name}")
    ensemble_metrics['learner_metrics'] = learner_metrics
    return ensemble_metrics
