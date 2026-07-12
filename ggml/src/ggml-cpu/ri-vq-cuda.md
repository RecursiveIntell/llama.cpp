# RI-VQ CUDA Scoring Kernel — Design Document

> **Status:** Design only. No local GPU for implementation. Written 2026-07-12.

## Problem

The CPU scoring loop is the bottleneck for proveKV at scale. The hot-path optimization reference proved the bottleneck is memory bandwidth, not compute. On CPU, the Gram table lookup loop is limited by L1 cache bandwidth and scattered key index access. On GPU, both are eliminated.

## Kernel Design

### Memory Layout

```
Gram table:     N×N f32 = 256×256×4 = 256KB (fits in shared memory on most GPUs)
Key indices:    n_tokens × block_count uint8 = n_tokens × 32 = 32*n_tokens bytes
Key norms:      n_tokens f32
Query indices:  block_count uint8 = 32 bytes
Query norm:     1 f32
```

### Kernel: `ri_vq_score_kernel`

```cuda
__global__ void ri_vq_score_kernel(
    const uint8_t* __restrict__ key_indices,  // [n_tokens * block_count]
    const float*   __restrict__ key_norms,    // [n_tokens]
    const float*   __restrict__ gram_rows,     // [block_count * N] (prefetched for this query)
    float query_norm,
    float scale,
    int block_count,
    int n_tokens,
    float* __restrict__ scores                 // [n_tokens] output
) {
    // Load prefetched Gram rows into shared memory (block_count * N * 4 bytes)
    // For block_count=32, N=256: 32KB — fits in shared memory
    extern __shared__ float s_gram_rows[];
    int tid = threadIdx.x;
    int row_size = block_count;  // each thread handles one token
    
    // Cooperative load of gram_rows into shared memory
    for (int i = tid; i < block_count * N; i += blockDim.x) {
        s_gram_rows[i] = gram_rows[i];
    }
    __syncthreads();
    
    // Each thread scores one token
    int token = blockIdx.x * blockDim.x + tid;
    if (token >= n_tokens) return;
    
    const uint8_t* key_block = &key_indices[token * block_count];
    float total = 0.0f;
    for (int b = 0; b < block_count; b++) {
        total += s_gram_rows[b * N + key_block[b]];
    }
    scores[token] = total * query_norm * key_norms[token] * scale;
}
```

### Kernel: `ri_vq_topk_reduction`

After scoring, select top-k tokens. Use a parallel bitonic sort or radix select in shared memory:

```cuda
__global__ void ri_vq_topk_kernel(
    const float* __restrict__ scores,
    int n_tokens,
    int k,
    int* __restrict__ top_indices,    // [k] output
    float* __restrict__ top_scores     // [k] output
) {
    // Block-level reduction: each block finds its local top-k
    // Then inter-block reduction to find global top-k
    // For k=64 and n_tokens=4096: 4096/256 = 16 blocks, each finds local top-64
    // Final merge: 16 × 64 = 1024 candidates, sort to get global top-64
}
```

### Kernel: `ri_vq_decode_selected_values`

```cuda
__global__ void ri_vq_decode_v_kernel(
    const uint8_t* __restrict__ value_indices,  // [n_tokens * block_count]
    const float*   __restrict__ value_norms,      // [n_tokens]
    const float*   __restrict__ codebook,         // [N * block_dim]
    int block_dim,
    int block_count,
    const int* __restrict__ selected,              // [k] token indices
    int k,
    float* __restrict__ decoded_values             // [k * head_dim] output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    int token = selected[idx];
    const uint8_t* v_block = &value_indices[token * block_count];
    float* out = &decoded_values[idx * block_count * block_dim];
    for (int b = 0; b < block_count; b++) {
        const float* codeword = &codebook[v_block[b] * block_dim];
        for (int d = 0; d < block_dim; d++) {
            out[b * block_dim + d] = codeword[d];
        }
    }
    // Scale by norm
    float norm = value_norms[token];
    for (int i = 0; i < block_count * block_dim; i++) {
        out[i] *= norm;
    }
}
```

### Launch Configuration

```
Scoring kernel:
  grid:   ceil(n_tokens / 256)
  block:  256 threads
  shared: block_count * N * sizeof(float) = 32 * 256 * 4 = 32KB
  
Top-k reduction:
  grid:   1 (single block for n_tokens < 8192) or multi-block
  block:  256 threads
  
Value decode:
  grid:   ceil(k / 32)
  block:  32 threads
```

### Expected Speedup

Based on CPU profiling:
- CPU scoring: 60μs for 2048 tokens (head_dim=128, block_dim=4)
- CPU bottleneck: L1 cache bandwidth for Gram table lookups
- GPU: Gram table in shared memory (32KB), key indices in global memory (coalesced reads)
- Expected: 10-50x for scoring, 5-10x end-to-end (including codebook training + V decode)

The GPU eliminates the memory bandwidth bottleneck that the CPU hot-path optimization reference identified as the limiting factor. The scoring loop is embarrassingly parallel — each token is independent.

### Integration Path

The CUDA kernel would replace the `score_all_tokens` call in ri-vq.cpp when CUDA is available. The flow:

1. CPU: Train codebook + build Gram table (once per layer)
2. CPU: Quantize K/V vectors (once per token)
3. GPU: Score all tokens for current query (per query position)
4. GPU: Top-k selection
5. GPU: Decode V for selected tokens only
6. CPU: Softmax-weighted V sum (or GPU if large k)

### Hardware Requirements

- GPU with ≥ 48KB shared memory per SM (for N=256, block_dim=4)
- CUDA compute capability ≥ 7.0 (Volta+) for optimal uint8 gather
- For N=256, block_dim=8: shared memory = 16KB (fits on all modern GPUs)

### Claim Boundary

- This is a design document, not a working implementation
- Speedup estimates are based on CPU profiling extrapolation, not measurement
- The kernel may need warp-level reductions or cooperative groups for optimal top-k
- Memory transfer overhead (K/V indices to GPU) is not accounted for and may dominate at small batch sizes