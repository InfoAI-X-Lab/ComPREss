"""Data preparation, Dataset classes, samplers, and DataLoader collate functions."""

import os
import random
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, Sampler
from data_augmentation import augment_class_data

def load_and_augment_data(csv_path, fold_idx, n_folds=5, target_count=100):
    dataset_dir = 'path/to/label.csv'
    train_indices_file = os.path.join(dataset_dir, f'fold_{fold_idx + 1}_train_original.csv')
    test_indices_file = os.path.join(dataset_dir, f'fold_{fold_idx + 1}_test.csv')
    augmented_train_file = os.path.join(dataset_dir, f'fold_{fold_idx + 1}_train_augmented.csv')
    if (os.path.exists(train_indices_file) and 
        os.path.exists(test_indices_file) and 
        os.path.exists(augmented_train_file)):
        print(f"Detected existing data files for Fold {fold_idx + 1}, loading directly...")
        train_df = pd.read_csv(augmented_train_file)
        test_df = pd.read_csv(test_indices_file)
        print(f"Fold {fold_idx + 1}: Loaded training set size: {len(train_df)}, test set size: {len(test_df)}")
        print(f"Loading from file: {augmented_train_file}")
        print(f"Loading from file: {test_indices_file}")
    else:
        print(f"Data files for Fold {fold_idx + 1} are incomplete, starting data reprocessing...")
        df = pd.read_csv(csv_path)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        folds = list(skf.split(df['sequence'], df['label']))
        train_idx, test_idx = folds[fold_idx]
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        print(f"Fold {fold_idx + 1}: Train size: {len(train_df)}, Test size: {len(test_df)}")
        os.makedirs(dataset_dir, exist_ok=True)
        train_indices_df = pd.DataFrame({
            'original_index': train_idx,
            'sequence': df.iloc[train_idx]['sequence'].values,
            'label': df.iloc[train_idx]['label'].values
        })
        test_indices_df = pd.DataFrame({
            'original_index': test_idx,
            'sequence': df.iloc[test_idx]['sequence'].values,
            'label': df.iloc[test_idx]['label'].values
        })
        train_indices_df.to_csv(train_indices_file, index=False)
        test_indices_df.to_csv(test_indices_file, index=False)
        print(f"Original training split saved to: {train_indices_file}")
        print(f"Test split saved to: {test_indices_file}")
        print("Starting data augmentation...")
        for class_label in range(6):
            train_df = augment_class_data(train_df, class_label, target_count)
        print(f"Training set size after data augmentation: {len(train_df)}")
        train_df.to_csv(augmented_train_file, index=False)
        print(f"Augmented training set saved to: {augmented_train_file}")
    train_class_sequences = {}
    test_class_sequences = {}
    for i in range(6):
        train_class_sequences[i] = train_df[train_df['label'] == i]['sequence'].tolist()
        test_class_sequences[i] = test_df[test_df['label'] == i]['sequence'].tolist()
    return train_class_sequences, test_class_sequences

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=1024, overlap=100):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        if len(sequence) > self.max_length:
            window_size = self.max_length - 2
            step_size = window_size - self.overlap
            windows = []
            for start in range(0, len(sequence) - window_size + 1, step_size):
                window = sequence[start:start + window_size]
                encoding = self.tokenizer(window,
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          padding="max_length",
                                          truncation=True,
                                          return_tensors="pt")
                windows.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                })
            input_ids = torch.stack([w['input_ids'] for w in windows])
            attention_mask = torch.stack([w['attention_mask'] for w in windows])
        else:
            encoding = self.tokenizer(sequence,
                                      add_special_tokens=True,
                                      max_length=self.max_length,
                                      padding="max_length",
                                      truncation=True,
                                      return_tensors="pt")
            input_ids = encoding['input_ids'].squeeze().unsqueeze(0)
            attention_mask = encoding['attention_mask'].squeeze().unsqueeze(0)
        return {
            'label': torch.tensor(label, dtype=torch.long),
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

class ContrastiveDataset(Dataset):
    def __init__(self, class_sequences, tokenizer, max_length=1024, overlap=100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        self.data = []
        for i in range(6):
            for seq in class_sequences[i]:
                self.data.append({
                    'sequence': seq,
                    'class_name': f'class_{i}',
                    'label': i
                })
        # Group data by class for easier negative sampling
        self.class_data = {}
        for i in range(6):
            self.class_data[f'class_{i}'] = class_sequences[i]
        print(f"Total sequences: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        original_seq = item['sequence']
        class_name = item['class_name']
        label = item['label']
        positive_seq = self.get_positive_sample(class_name, original_seq)
        anchor_features = self._process_sequence(original_seq)
        positive_features = self._process_sequence(positive_seq)
        return {
            'anchor': anchor_features,
            'positive': positive_features,
            'class_name': class_name,
            'label': label,
            'idx': idx
        }
    
    def _process_sequence(self, sequence):
        if len(sequence) <= self.max_length - 2:
            encoded = self.tokenizer(
                sequence,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
        windows = []
        step_size = self.max_length - self.overlap - 2
        for i in range(0, len(sequence), step_size):
            window_seq = sequence[i:i + self.max_length - 2]
            if len(window_seq) < 50:  
                break
            encoded = self.tokenizer(
                window_seq,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            windows.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })
        if not windows:
            encoded = self.tokenizer(
                sequence[:self.max_length - 2],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
        input_ids = torch.stack([w['input_ids'] for w in windows])
        attention_mask = torch.stack([w['attention_mask'] for w in windows])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def get_positive_sample(self, class_name, anchor_seq):
        sequences = self.class_data[class_name]
        if len(sequences) > 1:
            available_seqs = [seq for seq in sequences if seq != anchor_seq]
            if available_seqs:
                return random.choice(available_seqs)
        return anchor_seq

class TestDataset(Dataset):
    def __init__(self, class_sequences, tokenizer, max_length=1024, overlap=100):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        self.data = []
        for i in range(6):
            for seq in class_sequences[i]:
                self.data.append({
                    'sequence': seq,
                    'label': i
                })
        print(f"Test sequences: {len(self.data)}")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        sequence = item['sequence']
        label = item['label']
        features = self._process_sequence(sequence)
        return {
            'features': features,
            'label': label,
            'idx': idx
        }
    def _process_sequence(self, sequence):
        if len(sequence) <= self.max_length - 2:
            encoded = self.tokenizer(
                sequence,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
        windows = []
        step_size = self.max_length - self.overlap - 2
        for i in range(0, len(sequence), step_size):
            window_seq = sequence[i:i + self.max_length - 2]
            if len(window_seq) < 50:
                break
            encoded = self.tokenizer(
                window_seq,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            windows.append({
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            })
        if not windows:
            encoded = self.tokenizer(
                sequence[:self.max_length - 2],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0)
            }
        input_ids = torch.stack([w['input_ids'] for w in windows])
        attention_mask = torch.stack([w['attention_mask'] for w in windows])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_classes = 6
        self.class_indices = {f'class_{i}': [] for i in range(self.num_classes)}
        for idx, item in enumerate(dataset.data):
            self.class_indices[item['class_name']].append(idx)
        self.samples_per_class = batch_size // self.num_classes
        if self.samples_per_class == 0:
            raise ValueError(f"Batch size {batch_size} is too small for {self.num_classes} classes")
        remaining_samples = batch_size % self.num_classes
        self.class_samples = [self.samples_per_class] * self.num_classes
        for i in range(remaining_samples):
            self.class_samples[i] += 1
    
    def __iter__(self):
        class_shuffled = {}
        for i in range(self.num_classes):
            class_shuffled[f'class_{i}'] = self.class_indices[f'class_{i}'].copy()
            random.shuffle(class_shuffled[f'class_{i}'])
        max_batches = float('inf')
        for i in range(self.num_classes):
            class_name = f'class_{i}'
            if len(class_shuffled[class_name]) > 0:
                max_batches_for_class = len(class_shuffled[class_name]) // self.class_samples[i]
                max_batches = min(max_batches, max_batches_for_class)
        if max_batches == float('inf'):
            max_batches = 0
        for batch_idx in range(max_batches):
            batch = []
            for i in range(self.num_classes):
                class_name = f'class_{i}'
                start = batch_idx * self.class_samples[i]
                end = start + self.class_samples[i]
                batch.extend(class_shuffled[class_name][start:end])
            random.shuffle(batch)
            yield batch
    
    def __len__(self):
        max_batches = float('inf')
        for i in range(self.num_classes):
            class_name = f'class_{i}'
            if len(self.class_indices[class_name]) > 0:
                max_batches_for_class = len(self.class_indices[class_name]) // self.class_samples[i]
                max_batches = min(max_batches, max_batches_for_class)
        if max_batches == float('inf'):
            max_batches = 0
        return max_batches


def custom_collate_fn(batch):
    max_windows = max([item['input_ids'].size(0) for item in batch])
    batch_size = len(batch)
    input_ids = []
    attention_masks = []
    labels = []
    for item in batch:
        num_windows = item['input_ids'].size(0)
        if num_windows < max_windows:
            padding_windows = max_windows - num_windows
            pad_input_ids = torch.zeros((padding_windows, item['input_ids'].size(1)), dtype=item['input_ids'].dtype)
            pad_attention_mask = torch.zeros((padding_windows, item['attention_mask'].size(1)),
                                             dtype=item['attention_mask'].dtype)

            input_ids.append(torch.cat([item['input_ids'], pad_input_ids], dim=0))
            attention_masks.append(torch.cat([item['attention_mask'], pad_attention_mask], dim=0))
        else:
            input_ids.append(item['input_ids'])
            attention_masks.append(item['attention_mask'])
        labels.append(item['label'])
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'label': torch.stack(labels)
    }

def test_collate_fn(batch):
    max_windows = 1
    for item in batch:
        if item['features']['input_ids'].dim() > 1:
            max_windows = max(max_windows, item['features']['input_ids'].size(0))
    batch_size = len(batch)
    input_ids = []
    attention_masks = []
    labels = []
    indices = []
    for item in batch:
        input_ids_item = item['features']['input_ids']
        attention_mask_item = item['features']['attention_mask']
        if input_ids_item.dim() == 1:
            input_ids_item = input_ids_item.unsqueeze(0)
            attention_mask_item = attention_mask_item.unsqueeze(0)
        if input_ids_item.size(0) < max_windows:
            padding_windows = max_windows - input_ids_item.size(0)
            pad_input_ids = torch.zeros((padding_windows, input_ids_item.size(1)), dtype=input_ids_item.dtype)
            pad_attention_mask = torch.zeros((padding_windows, attention_mask_item.size(1)), dtype=attention_mask_item.dtype)
            input_ids_item = torch.cat([input_ids_item, pad_input_ids], dim=0)
            attention_mask_item = torch.cat([attention_mask_item, pad_attention_mask], dim=0)
        input_ids.append(input_ids_item)
        attention_masks.append(attention_mask_item)
        labels.append(item['label'])
        indices.append(item['idx'])
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'labels': labels,
        'indices': indices
    }

def contrastive_collate_fn(batch):
    max_windows_anchor = 1
    max_windows_pos = 1
    
    for item in batch:
        if item['anchor']['input_ids'].dim() > 1:
            max_windows_anchor = max(max_windows_anchor, item['anchor']['input_ids'].size(0))
        if item['positive']['input_ids'].dim() > 1:
            max_windows_pos = max(max_windows_pos, item['positive']['input_ids'].size(0))
    
    max_windows = max(max_windows_anchor, max_windows_pos)
    batch_size = len(batch)
    anchor_input_ids = []
    anchor_attention_masks = []
    positive_input_ids = []
    positive_attention_masks = []
    negative_input_ids = []
    negative_attention_masks = []
    class_names = []
    labels = []
    indices = []
    for item in batch:
        current_class = item['class_name']
        negative_samples = [other_item for other_item in batch if other_item['class_name'] != current_class]
        assert negative_samples, f"No negative samples found for class {current_class}."
        neg_item = random.choice(negative_samples)
        neg_features = neg_item['anchor']
        anchor_input_ids_item = item['anchor']['input_ids']
        anchor_attention_mask_item = item['anchor']['attention_mask']
        if anchor_input_ids_item.dim() == 1:
            anchor_input_ids_item = anchor_input_ids_item.unsqueeze(0)
            anchor_attention_mask_item = anchor_attention_mask_item.unsqueeze(0)
        if anchor_input_ids_item.size(0) < max_windows:
            padding_windows = max_windows - anchor_input_ids_item.size(0)
            pad_input_ids = torch.zeros((padding_windows, anchor_input_ids_item.size(1)), dtype=anchor_input_ids_item.dtype)
            pad_attention_mask = torch.zeros((padding_windows, anchor_attention_mask_item.size(1)), dtype=anchor_attention_mask_item.dtype)
            anchor_input_ids_item = torch.cat([anchor_input_ids_item, pad_input_ids], dim=0)
            anchor_attention_mask_item = torch.cat([anchor_attention_mask_item, pad_attention_mask], dim=0)
        pos_input_ids = item['positive']['input_ids']
        pos_attention_mask = item['positive']['attention_mask']
        if pos_input_ids.dim() == 1:
            pos_input_ids = pos_input_ids.unsqueeze(0)
            pos_attention_mask = pos_attention_mask.unsqueeze(0)
        if pos_input_ids.size(0) < max_windows:
            padding_windows = max_windows - pos_input_ids.size(0)
            pad_input_ids = torch.zeros((padding_windows, pos_input_ids.size(1)), dtype=pos_input_ids.dtype)
            pad_attention_mask = torch.zeros((padding_windows, pos_attention_mask.size(1)), dtype=pos_attention_mask.dtype)
            
            pos_input_ids = torch.cat([pos_input_ids, pad_input_ids], dim=0)
            pos_attention_mask = torch.cat([pos_attention_mask, pad_attention_mask], dim=0)
        neg_input_ids_item = neg_features['input_ids']
        neg_attention_mask_item = neg_features['attention_mask']
        if neg_input_ids_item.dim() == 1:
            neg_input_ids_item = neg_input_ids_item.unsqueeze(0)
            neg_attention_mask_item = neg_attention_mask_item.unsqueeze(0)
        if neg_input_ids_item.size(0) < max_windows:
            padding_windows = max_windows - neg_input_ids_item.size(0)
            pad_input_ids = torch.zeros((padding_windows, neg_input_ids_item.size(1)), dtype=neg_input_ids_item.dtype)
            pad_attention_mask = torch.zeros((padding_windows, neg_attention_mask_item.size(1)), dtype=neg_attention_mask_item.dtype)
            neg_input_ids_item = torch.cat([neg_input_ids_item, pad_input_ids], dim=0)
            neg_attention_mask_item = torch.cat([neg_attention_mask_item, pad_attention_mask], dim=0)
        anchor_input_ids.append(anchor_input_ids_item)
        anchor_attention_masks.append(anchor_attention_mask_item)
        positive_input_ids.append(pos_input_ids)
        positive_attention_masks.append(pos_attention_mask)
        negative_input_ids.append(neg_input_ids_item)
        negative_attention_masks.append(neg_attention_mask_item)
        class_names.append(item['class_name'])
        labels.append(item['label'])
        indices.append(item['idx'])
    return {
        'anchor': {
            'input_ids': torch.stack(anchor_input_ids),
            'attention_mask': torch.stack(anchor_attention_masks)
        },
        'positive': {
            'input_ids': torch.stack(positive_input_ids),
            'attention_mask': torch.stack(positive_attention_masks)
        },
        'negatives': {
            'input_ids': torch.stack(negative_input_ids),
            'attention_mask': torch.stack(negative_attention_masks)
        },
        'class_names': class_names,
        'labels': labels,
        'indices': indices
    }
