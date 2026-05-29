"""Loss functions for contrastive learning and class centers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedClassAwareSupConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(EnhancedClassAwareSupConLoss, self).__init__()
        self.temperature = temperature
    def forward(self, features, labels=None, mask=None):
        device = features.device
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        features = F.normalize(features, p=2, dim=2)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        if labels is not None:
            expanded_labels = labels.repeat(anchor_count, 1).view(-1)
            contrast_labels = labels.repeat(contrast_count, 1).view(-1)
            negative_enhancement = torch.ones_like(logits)
            clean_classes = [0, 1, 3, 4, 5]
            dirty_class = 2
            for clean_anchor_label in clean_classes:
                anchor_mask = (expanded_labels == clean_anchor_label).unsqueeze(1).expand(-1, len(contrast_labels))
                neg_dirty_mask = (contrast_labels == dirty_class).unsqueeze(0).expand(len(expanded_labels), -1)
                negative_enhancement[anchor_mask & neg_dirty_mask] = 5.0
                for other_clean_neg_label in clean_classes:
                    if clean_anchor_label == other_clean_neg_label:
                        continue
                    neg_other_clean_mask = (contrast_labels == other_clean_neg_label).unsqueeze(0).expand(len(expanded_labels), -1)
                    negative_enhancement[anchor_mask & neg_other_clean_mask] = 2.0
            anchor_dirty_mask = (expanded_labels == dirty_class).unsqueeze(1).expand(-1, len(contrast_labels))
            for clean_neg_label in clean_classes:
                neg_clean_mask = (contrast_labels == clean_neg_label).unsqueeze(0).expand(len(expanded_labels), -1)
                negative_enhancement[anchor_dirty_mask & neg_clean_mask] = 5.0
            positive_mask = mask
            negative_mask = (1 - mask) * logits_mask
            exp_logits = torch.exp(logits)
            positive_exp = exp_logits * positive_mask
            negative_exp = exp_logits * negative_mask * negative_enhancement
            total_exp = positive_exp + negative_exp
            log_prob = logits - torch.log(total_exp.sum(1, keepdim=True) + 1e-8)
        else:
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, alpha=0.01, exclude_class=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.exclude_class = exclude_class
        self.register_buffer('centers', torch.randn(num_classes, feature_dim))
    def forward(self, features, labels):
        batch_size = features.size(0)
        mask = labels != self.exclude_class
        if not mask.any():
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        filtered_features = features[mask]
        filtered_labels = labels[mask]
        centers_batch = self.centers[filtered_labels]
        loss = torch.sum((filtered_features - centers_batch) ** 2) / (2.0 * filtered_features.size(0))
        return loss
    def update_centers(self, features, labels):
        mask = labels != self.exclude_class
        if not mask.any():
            return
        filtered_features = features[mask]
        filtered_labels = labels[mask]
        for class_id in torch.unique(filtered_labels):
            class_mask = filtered_labels == class_id
            if class_mask.any():
                class_features = filtered_features[class_mask]
                class_center = torch.mean(class_features, dim=0)
                self.centers[class_id] = (1 - self.alpha) * self.centers[class_id] + self.alpha * class_center

