// ri-vq.h — RecursiveIntell vector-quantization compressed KV cache attention
//
// This implements the proveKV algorithm: block-based vector quantization with
// a pre-computed Gram table enabling compressed-domain attention scoring without
// full decompression of K vectors.
//
// Supports two codebook sizes:
//   N=256  (uint8 indices, 256KB Gram table, block_dim=4: 7.1x compression)
//   N=1024 (uint16 indices, 4MB Gram table, block_dim=8: 12.8x compression)
//
// Ported from Rust proveKV implementation (poly-kv/src/pool.rs).
// Reference: https://github.com/RecursiveIntell/Libraries

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace ri_vq {

// Configuration
constexpr int    DEFAULT_BLOCK_DIM    = 4;
constexpr int    MAX_CODEBOOK_SIZE    = 1024;
constexpr int    MAX_BLOCK_DIM        = 128;

// Index type: uint8 for N<=256, uint16 for N<=1024
using index_t = uint16_t;

// Compressed key vector: codebook indices + L2 norm
struct CompressedKey {
    index_t indices[MAX_BLOCK_DIM];
    float   norm;
    int     n_blocks;

    size_t byte_size() const {
        return n_blocks * sizeof(index_t) + sizeof(float);
    }
};

// Compressed value vector: same encoding as keys
struct CompressedValue {
    index_t indices[MAX_BLOCK_DIM];
    float   norm;
    int     n_blocks;
};

// Codebook: N codewords of dimension BLOCK_DIM
// Heap-allocated to support N=1024 (1024*128*4 = 512KB)
struct Codebook {
    std::vector<float> table;   // N * BLOCK_DIM
    int   n;
    int   block_dim;

    Codebook() : n(0), block_dim(0) {}

    void init(int n_, int block_dim_) {
        n = n_;
        block_dim = block_dim_;
        table.resize((size_t)n * block_dim);
    }

    const float* codeword(int i) const {
        return &table[(size_t)i * block_dim];
    }
    float* codeword(int i) {
        return &table[(size_t)i * block_dim];
    }
};

// Pre-computed Gram table: G[i,j] = dot(codebook[i], codebook[j])
// Heap-allocated: N=256 -> 256KB, N=1024 -> 4MB
struct GramTable {
    std::vector<float> values;   // N*N
    int   n;
    int   block_dim;

    GramTable() : n(0), block_dim(0) {}

    void init(int n_) {
        n = n_;
        values.resize((size_t)n * n);
    }

    float at(int i, int j) const {
        return values[(size_t)i * n + j];
    }
    float& at(int i, int j) {
        return values[(size_t)i * n + j];
    }
};

// Prepared query: quantized query blocks + query norm
struct PreparedQuery {
    index_t query_indices[MAX_BLOCK_DIM];
    float   query_norm;
    int     n_blocks;
};

// Prefetched Gram rows: block_count rows copied into contiguous buffer
struct PrefetchedGramRows {
    std::vector<float> gram_rows;
    int   n;
    int   block_count;
    float query_norm;
};

// Pre-fetched compressed index
struct FullyPreparedIndex {
    std::vector<index_t> key_indices_flat;
    std::vector<float>   key_norms;
    int n_blocks;
    int n;
    int block_dim;
    int head_dim;
    int num_tokens;

    const index_t* key_block(int token_idx) const {
        return &key_indices_flat[(size_t)token_idx * n_blocks];
    }
};

// ---- Core API ----

Codebook train_codebook(const float* samples, int n_samples, int block_dim,
                        int codebook_size, int max_iter, uint64_t seed);

GramTable build_gram_table(const Codebook& cb);

int quantize_block(const float* block, const Codebook& cb);

CompressedKey quantize_key(const float* key, int head_dim, const Codebook& cb);
CompressedValue quantize_value(const float* value, int head_dim, const Codebook& cb);

void decode_key(const CompressedKey& ck, const Codebook& cb, float* out, int head_dim);
void decode_value(const CompressedValue& cv, const Codebook& cb, float* out, int head_dim);

PreparedQuery prepare_query(const float* query, int head_dim, const Codebook& cb);
PrefetchedGramRows prefetch_gram_rows(const PreparedQuery& pq, const GramTable& gram);

FullyPreparedIndex build_prepared_index(const std::vector<CompressedKey>& keys,
                                          int num_tokens, int head_dim,
                                          const Codebook& cb);

std::vector<std::pair<int, float>> score_all_tokens(
    const FullyPreparedIndex& index,
    const PrefetchedGramRows& prefetched);

struct AttentionHit {
    int   token_index;
    float score;
    std::vector<float> value;
};

std::vector<AttentionHit> compressed_attention_topk(
    const FullyPreparedIndex& k_index,
    const std::vector<CompressedValue>& v_codes,
    const Codebook& cb,
    const float* query,
    int head_dim,
    int top_k);

std::vector<std::vector<AttentionHit>> compressed_attention_topk_batch(
    const FullyPreparedIndex& k_index,
    const std::vector<CompressedValue>& v_codes,
    const Codebook& cb,
    const float* const* queries,
    int num_heads,
    int head_dim,
    int top_k);

float compression_ratio(int head_dim, int block_dim, int codebook_size);
size_t compressed_size_bytes(int head_dim, int block_dim, int codebook_size);
size_t f16_size_bytes(int head_dim);

} // namespace ri_vq