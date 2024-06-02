

import torch
from torch import nn, Tensor
from typing import Tuple
from torch.utils.data import dataset


def data_process_webtext(raw_text, _tokenizer, start_id = 0, end_id = 10000):
        
    data = []
    for _id, item in enumerate(raw_text['text']):
        if _id >= start_id: data.append(torch.tensor(_tokenizer(item)['input_ids'], dtype=torch.long))
        if _id == end_id: break                                   
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def data_process(raw_text_iter, _tokenizer):
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(_tokenizer(item)['input_ids'], dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data, bsz, device):

    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, i: int, bptt: int) -> Tuple[Tensor, Tensor]:

    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target