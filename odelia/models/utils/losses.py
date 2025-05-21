import torch
import torch.nn as nn
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

import torch.nn.functional as F

class CornLossMulti(torch.nn.Module):
    """
    Compute the CORN loss for multi-class classification.
    """
    def __init__(self, class_labels_num):
        super().__init__()
        self.class_labels_num = class_labels_num # [Classes, Labels]
    
    def forward(self, logits, targets):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, num_classes*(num_labels-1)]
            targets: torch.Tensor, shape [batch_size]
        """
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        loss = 0
        for c, chunk in enumerate(chunks):
            loss += corn_loss(chunk, targets[:, c], chunk.shape[1]+1)
        return loss/len(chunks)

    def logits2labels(self, logits):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, num_classes*(num_labels-1)]
        """
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        labels = []
        for c, chunk in enumerate(chunks):
            label = corn_label_from_logits(chunk)
            labels.append(label)
        return torch.stack(labels, dim=1)
    
    def logits2probabilities(self, logits):
        # Argmax can leed to different output: https://github.com/Raschka-research-group/coral-pytorch/discussions/27
        """
        Args:
            logits: torch.Tensor, shape [batch_size, num_classes*(num_labels-1)]
        """
        chunks = torch.split(logits, self.class_labels_num, dim=1)
        classes_probs = []
        for c, chunk in enumerate(chunks):
            cumulative_probs = torch.sigmoid(chunk)
            # cumulative_probs = torch.cumprod(probas, dim=1)
            
            # Add boundary conditions P(y >= 1) = 1 and P(y >= num_classes) = 0
            cumulative_probs = torch.cat([torch.ones_like(cumulative_probs[:, :1]), cumulative_probs, torch.zeros_like(cumulative_probs[:, :1])], dim=1)
            
            # Compute class probabilities
            # cumulative_probs = torch.cat([torch.ones_like(cumulative_probs[:, :1]), cumulative_probs], dim=1)
            probs = cumulative_probs[:, :-1] - cumulative_probs[:, 1:]
            # probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-6)
            # probs = cumulative_probs
          
            classes_probs.append(probs)
        return torch.stack(classes_probs, dim=1)
    




class MulitCELoss(nn.Module):
    """
    CrossEntropyLoss per class-label group.
    """
    def __init__(self, class_labels_num):
        """
        Args:
            class_labels_num: List[int], number of labels for each class group
        """
        super().__init__()
        self.class_labels_num = class_labels_num
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, sum(class_labels_num)]
            targets: torch.Tensor, shape [batch_size, len(class_labels_num)]
        """
        chunks_logits = torch.split(logits, self.class_labels_num, dim=1)
        loss = 0
        for i, logit_chunk in enumerate(chunks_logits):
            target_chunk = targets[:, i]
            loss += self.criterion(logit_chunk, target_chunk)

        return loss / len(chunks_logits)

    def logits2labels(self, logits):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, sum(class_labels_num)]
        Returns:
            torch.Tensor, shape [batch_size, len(class_labels_num)]
        """
        chunks_logits = torch.split(logits, self.class_labels_num, dim=1)
        labels = [torch.argmax(chunk, dim=1) for chunk in chunks_logits]
        return torch.stack(labels, dim=1)

    def logits2probabilities(self, logits):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, sum(class_labels_num)]
        Returns:
            torch.Tensor, shape [batch_size, len(class_labels_num), max_labels]
        """
        chunks_logits = torch.split(logits, self.class_labels_num, dim=1)
        probs = [F.softmax(chunk, dim=1) for chunk in chunks_logits]
        return torch.stack(probs, dim=1)


class MultiBCELoss(nn.Module):
    """
        BCEWithLogitsLoss per class-label group.
    """
    def __init__(self, class_labels_num):
        """
        Args:
            class_labels_num: List[int], number of labels for each class group
        """
        super().__init__()
        self.class_labels_num = class_labels_num
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: torch.Tensor, shape [batch_size, sum(class_labels_num)]
            targets: torch.Tensor, shape [batch_size, sum(class_labels_num)]
        """
        chunks_logits = torch.split(logits, self.class_labels_num, dim=1)
        chunks_targets = torch.split(targets, self.class_labels_num, dim=1)

        loss = 0
        for logit_chunk, target_chunk in zip(chunks_logits, chunks_targets):
            loss += self.criterion(logit_chunk, target_chunk.float())

        return loss / len(chunks_logits)

    def logits2labels(self, logits, threshold=0.5):
        probs = torch.sigmoid(logits)
        return (probs > threshold).int()

    def logits2probabilities(self, logits):
        return torch.sigmoid(logits)