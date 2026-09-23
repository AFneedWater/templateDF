"""Masked 20-class reconstruction loss and correctly weighted metric totals."""

import math

import torch
from torch.nn import functional as F

from .data import PAD_LABEL


def reconstruction_loss(logits, reconstructed_embeddings, target_embeddings,
                        labels, valid_mask, *, lambda_emb=0.0):
    if not math.isfinite(lambda_emb) or lambda_emb < 0:
        raise ValueError("lambda_emb must be finite and nonnegative")
    if labels.ndim != 2 or labels.dtype != torch.long:
        raise ValueError("labels must be int64 [B,L]")
    if valid_mask.shape != labels.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool [B,L]")
    if labels.shape[0] == 0 or not valid_mask.any(1).all():
        raise ValueError("Every sequence must contain valid residues")
    if logits.shape != (*labels.shape, 20):
        raise ValueError("logits must be [B,L,20]")
    if (target_embeddings.ndim != 3 or target_embeddings.shape[:2] != labels.shape
            or reconstructed_embeddings.shape != target_embeddings.shape
            or target_embeddings.shape[-1] == 0):
        raise ValueError("Embedding targets and predictions must match [B,L,D]")
    tensors=(logits,reconstructed_embeddings,target_embeddings,labels,valid_mask)
    if any(t.device != logits.device for t in tensors):
        raise ValueError("Loss inputs must share a device")
    if not all(t.is_floating_point() for t in tensors[:3]):
        raise ValueError("Logits and embeddings must be floating tensors")
    if ((labels[valid_mask]<0)|(labels[valid_mask]>=20)).any() or (labels[~valid_mask]!=PAD_LABEL).any():
        raise ValueError("Valid labels must be 0..19 and padding labels must be -100")
    # Select before arithmetic: even NaN/Inf in masked positions cannot pollute loss.
    scores=logits[valid_mask].float()
    predicted=reconstructed_embeddings[valid_mask].float()
    target=target_embeddings[valid_mask].float()
    if not all(torch.isfinite(t).all() for t in (scores,predicted,target)):
        raise ValueError("Valid loss inputs must be finite")
    ce_sum=F.cross_entropy(scores,labels[valid_mask],ignore_index=PAD_LABEL,reduction='sum')
    mse_sum=(predicted-target).square().mean(-1).sum()
    residue_count=int(valid_mask.sum().item())
    ce=ce_sum/residue_count
    mse=mse_sum/residue_count
    correct=(logits.detach().argmax(-1)==labels)&valid_mask
    sequence_accuracy_sum=(correct.sum(1).float()/valid_mask.sum(1)).sum()
    return {'loss':ce+lambda_emb*mse,'ce':ce,'embedding_mse':mse,
            'residue_accuracy':correct.sum().float()/residue_count,
            'sequence_accuracy':sequence_accuracy_sum/labels.shape[0],
            'ce_sum':ce_sum.detach(),'embedding_mse_sum':mse_sum.detach(),
            'correct_residues':int(correct.sum().item()),
            'sequence_accuracy_sum':sequence_accuracy_sum.detach(),
            'residue_count':residue_count,'sequence_count':labels.shape[0]}


class MetricAccumulator:
    def __init__(self):
        self.ce_sum=self.mse_sum=self.sequence_accuracy_sum=0.0
        self.correct=self.residues=self.sequences=0

    def update(self, metrics):
        self.ce_sum+=float(metrics['ce_sum'])
        self.mse_sum+=float(metrics['embedding_mse_sum'])
        self.sequence_accuracy_sum+=float(metrics['sequence_accuracy_sum'])
        self.correct+=metrics['correct_residues']
        self.residues+=metrics['residue_count']
        self.sequences+=metrics['sequence_count']

    def compute(self, lambda_emb):
        if not self.residues or not self.sequences:
            raise ValueError("Cannot report metrics for an empty dataset")
        ce=self.ce_sum/self.residues
        mse=self.mse_sum/self.residues
        return {'loss':ce+lambda_emb*mse,'ce':ce,'embedding_mse':mse,
                'residue_accuracy':self.correct/self.residues,
                'sequence_accuracy':self.sequence_accuracy_sum/self.sequences,
                'residue_count':self.residues,'sequence_count':self.sequences}
