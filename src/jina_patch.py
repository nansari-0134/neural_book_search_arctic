"""
Jina Reranker v2 Compatibility Patch
====================================
Patches transformers 5.x compatibility for jina-reranker-v2-base-multilingual
by ensuring create_position_ids_from_input_ids is registered in xlm_roberta.
"""

import torch
import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm_mod

def _create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx

if not hasattr(xlm_mod, "create_position_ids_from_input_ids"):
    xlm_mod.create_position_ids_from_input_ids = _create_position_ids_from_input_ids
