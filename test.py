import torch
def get_attn_pad_mask(seq_q, seq_k, dec=False):
    batch, len_q = seq_q.size()
    batch, len_k = seq_k.size()
   
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) # [batch, 1, len_k]
    print(pad_attn_mask)
    return pad_attn_mask.expand(batch, len_q, len_k) # [batch, len_q, len_k]

seq_q = torch.tensor([[4, 1, 2, 3, 0], [5, 6, 7, 0, 0]]) 
seq_k = torch.tensor([[1, 2, 0, 0, 0], [2, 1, 7, 3, 0]])

pad_attn_mask = get_attn_pad_mask(seq_q, seq_k)
print(pad_attn_mask)