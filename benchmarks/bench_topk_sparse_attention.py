"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F
import numpy as np

from flashinfer.testing.utils import bench_gpu_time

def naive_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False):
    batch_size = len(cu_seqlens_q) - 1
    max_seqlen_q = max(cu_seqlens_q[i+1] - cu_seqlens_q[i] for i in range(batch_size))
    max_seqlen_k = max(cu_seqlens_k[i+1] - cu_seqlens_k[i] for i in range(batch_size))
    
    q_padded = torch.zeros(q.shape[0], batch_size, max_seqlen_q, q.shape[-1], device=q.device, dtype=q.dtype)
    k_padded = torch.zeros(k.shape[0], batch_size, max_seqlen_k, k.shape[-1], device=k.device, dtype=k.dtype)
    v_padded = torch.zeros(v.shape[0], batch_size, max_seqlen_k, v.shape[-1], device=v.device, dtype=v.dtype)
    
    for i in range(batch_size):
        q_start = cu_seqlens_q[i]
        q_end = cu_seqlens_q[i + 1]
        k_start = cu_seqlens_k[i]
        k_end = cu_seqlens_k[i + 1]
        q_padded[:, i, :q_end-q_start] = q[:, q_start:q_end]
        k_padded[:, i, :k_end-k_start] = k[:, k_start:k_end]
        v_padded[:, i, :k_end-k_start] = v[:, k_start:k_end]
    
    k_padded = k_padded.repeat_interleave(q_padded.shape[0] // k_padded.shape[0], dim=0)
    v_padded = v_padded.repeat_interleave(q_padded.shape[0] // v_padded.shape[0], dim=0)
    
    attn = q_padded @ k_padded.transpose(-2, -1) / (q_padded.size(-1) ** 0.5)
    
    if causal:
        causal_mask = torch.zeros(batch_size, max_seqlen_q, max_seqlen_k, device=q.device).bool()
        for b in range(batch_size):
            for i in range(max_seqlen_q):
                for j in range(max_seqlen_k):
                    if i >= (j * 16 + 32 - 1):
                        causal_mask[b, i, j] = True
        attn = attn.masked_fill(~causal_mask, -float('inf'))
    
    score = F.softmax(attn, dim=-1)
    score = score.reshape(2, 16, batch_size, max_seqlen_q, max_seqlen_k).sum(dim=1)
    
    result = []
    for i in range(batch_size):
        q_start = cu_seqlens_q[i]
        q_end = cu_seqlens_q[i + 1]
        k_start = cu_seqlens_k[i]
        k_end = cu_seqlens_k[i + 1]
        
        curr_score = torch.full((2, q_end-q_start, max_seqlen_k), 0, device=q.device, dtype=q.dtype)
        curr_score[:, :, :k_end-k_start] = score[:, i, :q_end-q_start, :k_end-k_start]
        result.append(curr_score)
    
    final_result = torch.cat(result, dim=1)
    final_result = torch.where(torch.isnan(final_result), 0, final_result)
    return final_result

def bench_topk_sparse_attention(
    num_qo_heads,
    num_kv_heads,
    head_dim,
    seqlen_q,
    seqlen_k,
    batch_size,
    dtype
):
    total_seqlen_q = batch_size * seqlen_q
    total_seqlen_k = batch_size * seqlen_k
    
    q = torch.randn(num_qo_heads, total_seqlen_q, head_dim, dtype=dtype).cuda()
    k = torch.randn(num_kv_heads, total_seqlen_k, head_dim, dtype=dtype).cuda()
    v = torch.randn(num_kv_heads, total_seqlen_k, head_dim, dtype=dtype).cuda()

    cu_seqlens_q = torch.arange(0, total_seqlen_q + 1, seqlen_q, device='cuda', dtype=torch.int32)
    cu_seqlens_k = torch.arange(0, total_seqlen_k + 1, seqlen_k, device='cuda', dtype=torch.int32)

    naive_implementation_time = np.median(
            bench_gpu_time(
                lambda: naive_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=True),
                dry_run_iters=2,
                repeat_iters=2,
            )
        )
    print(f"TopK Sparse Attention (naive) - Batch Size: {batch_size}, SeqLen Q: {seqlen_q}, SeqLen K: {seqlen_k}, Heads QO: {num_qo_heads}, Heads KV: {num_kv_heads}, Head Dim: {head_dim}, Dtype: {dtype} -> Time: {naive_implementation_time:.2f} ms")


if __name__ == "__main__":
    for batch_size in [2, 4, 8]: 
        for seqlen_q in [128, 256, 512]:
            for seqlen_k in [8, 16, 32]:
                for num_qo_heads in [32]:
                    for num_kv_heads in [32]:
                        for head_dim in [128]:
                            for dtype in [torch.half, torch.bfloat16, torch.float32]:
                                bench_topk_sparse_attention(
                                    num_qo_heads,
                                    num_kv_heads,
                                    head_dim,
                                    seqlen_q,
                                    seqlen_k,
                                    batch_size,
                                    dtype
                                )
