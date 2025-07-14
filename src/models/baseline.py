from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.tokenization_utils_base import BatchEncoding
from transformers.modeling_utils import PreTrainedModel
from typing import Dict


class SimpleBertClassifier(nn.Module):
    """
    A light‑weight classifier that puts a shallow MLP 
    on top of BERT’s pooled output.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        num_classes: int = 2
    ) -> None:
        super().__init__()

        # backbone
        self.bert: PreTrainedModel = AutoModel.from_pretrained(model_name)

        # classifier head
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self,
                x: BatchEncoding | Dict[str, torch.Tensor]
                ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : BatchEncoding | Dict[str, torch.Tensor]
            A “batch” from the NewsDataLoader; must contain at least
            `input_ids` and `attention_mask` (and optionally `token_type_ids`).

        Returns
        -------
        torch.Tensor
            Raw (unnormalised) logits of shape (batch_size, num_classes).
        """
        # AutoModel accepts **dict ; BatchEncoding is a dict subclass
        outputs = self.bert(**x)
        pooled: torch.Tensor = outputs.pooler_output

        h = self.dropout(pooled)
        h = self.fc(h)
        return h
