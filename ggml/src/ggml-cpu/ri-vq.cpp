// ri-vq.cpp — RecursiveIntell vector-quantization compressed KV cache attention
//
// Implementation of the proveKV algorithm: block-based VQ with Gram table
// for compressed-domain attention scoring.
//
// Ported from Rust proveKV (poly-kv/src/pool.rs).
// Reference: https://github.com/RecursiveIntell/Libraries

#include "ri-vq.h"
#include <random>
#include <limits>
#include <chrono>

namespace ri_vq {

// ---- Codebook training (simple k-means) ----

Codebook train_codebook(const float* samples, int n_samples, int block_dim,
                        int codebook_size, int max_iter, uint64_t seed) {
    Codebook cb;
    cb.n = codebook_size;
    cb.block_dim = block_dim;

    // Handle edge cases
    if (n_samples == 0 || block_dim <= 0 || codebook_size <= 0) {
        cb.n = 0;
        return cb;
    }

    // Initialize: k-means++ for better initial centroid coverage
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, n_samples - 1);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    int actual_size = std::min(codebook_size, n_samples);
    cb.n = actual_size;

    // k-means++: first centroid is random, rest are chosen with probability
    // proportional to squared distance from nearest existing centroid
    std::vector<float> min_dist_sq(n_samples, std::numeric_limits<float>::max());

    // Pick first centroid randomly
    int first_idx = dist(rng);
    std::memcpy(cb.codeword(0), &samples[first_idx * block_dim], block_dim * sizeof(float));

    for (int i = 1; i < actual_size; i++) {
        // Update min distances to the newly added centroid
        const float* centroid = cb.codeword(i - 1);
        float total_dist = 0.0f;
        for (int s = 0; s < n_samples; s++) {
            const float* sample = &samples[s * block_dim];
            float d_sq = 0.0f;
            for (int d = 0; d < block_dim; d++) {
                float diff = sample[d] - centroid[d];
                d_sq += diff * diff;
            }
            if (d_sq < min_dist_sq[s]) min_dist_sq[s] = d_sq;
            total_dist += min_dist_sq[s];
        }
        // Pick next centroid with probability proportional to distance
        if (total_dist <= 0.0f) {
            // All samples are identical to existing centroids, pick random
            int idx = dist(rng);
            std::memcpy(cb.codeword(i), &samples[idx * block_dim], block_dim * sizeof(float));
        } else {
            float r = uniform(rng) * total_dist;
            float cumsum = 0.0f;
            int chosen = n_samples - 1;
            for (int s = 0; s < n_samples; s++) {
                cumsum += min_dist_sq[s];
                if (cumsum >= r) { chosen = s; break; }
            }
            std::memcpy(cb.codeword(i), &samples[chosen * block_dim], block_dim * sizeof(float));
        }
    }

    // K-means iterations
    std::vector<int> assignments(n_samples, 0);
    std::vector<int> counts(actual_size, 0);
    std::vector<float> new_centroids(actual_size * block_dim, 0.0f);

    for (int iter = 0; iter < max_iter; iter++) {
        // Assign each sample to nearest centroid
        bool changed = false;
        for (int s = 0; s < n_samples; s++) {
            const float* sample = &samples[s * block_dim];
            float best_dist = std::numeric_limits<float>::max();
            int best_idx = 0;
            for (int c = 0; c < actual_size; c++) {
                float dist_sq = 0.0f;
                const float* centroid = cb.codeword(c);
                for (int d = 0; d < block_dim; d++) {
                    float diff = sample[d] - centroid[d];
                    dist_sq += diff * diff;
                }
                if (dist_sq < best_dist) {
                    best_dist = dist_sq;
                    best_idx = c;
                }
            }
            if (assignments[s] != best_idx) {
                assignments[s] = best_idx;
                changed = true;
            }
        }

        if (!changed && iter > 0) break;

        // Update centroids
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);

        for (int s = 0; s < n_samples; s++) {
            int c = assignments[s];
            counts[c]++;
            for (int d = 0; d < block_dim; d++) {
                new_centroids[c * block_dim + d] += samples[s * block_dim + d];
            }
        }

        for (int c = 0; c < actual_size; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < block_dim; d++) {
                    new_centroids[c * block_dim + d] /= counts[c];
                }
                std::memcpy(cb.codeword(c), &new_centroids[c * block_dim], block_dim * sizeof(float));
            }
            // Empty clusters: reinitialize to a random sample
            if (counts[c] == 0) {
                int idx = dist(rng);
                std::memcpy(cb.codeword(c), &samples[idx * block_dim], block_dim * sizeof(float));
            }
        }
    }

    return cb;
}

// ---- Gram table construction ----

GramTable build_gram_table(const Codebook& cb) {
    GramTable gram;
    gram.n = cb.n;
    gram.block_dim = cb.block_dim;

    for (int i = 0; i < cb.n; i++) {
        for (int j = 0; j < cb.n; j++) {
            float dot = 0.0f;
            const float* ci = cb.codeword(i);
            const float* cj = cb.codeword(j);
            for (int d = 0; d < cb.block_dim; d++) {
                dot += ci[d] * cj[d];
            }
            gram.at(i, j) = dot;
        }
    }

    return gram;
}

// ---- Quantization ----

int quantize_block(const float* block, const Codebook& cb) {
    float best_dist = std::numeric_limits<float>::max();
    int best_idx = 0;
    for (int c = 0; c < cb.n; c++) {
        float dist_sq = 0.0f;
        const float* centroid = cb.codeword(c);
        for (int d = 0; d < cb.block_dim; d++) {
            float diff = block[d] - centroid[d];
            dist_sq += diff * diff;
        }
        if (dist_sq < best_dist) {
            best_dist = dist_sq;
            best_idx = c;
        }
    }
    return best_idx;
}

CompressedKey quantize_key(const float* key, int head_dim, const Codebook& cb) {
    CompressedKey ck;
    ck.n_blocks = head_dim / cb.block_dim;
    ck.norm = 0.0f;

    // Compute L2 norm
    for (int d = 0; d < head_dim; d++) {
        ck.norm += key[d] * key[d];
    }
    ck.norm = std::sqrt(ck.norm);

    // Quantize each block
    for (int b = 0; b < ck.n_blocks; b++) {
        ck.indices[b] = (uint8_t)quantize_block(&key[b * cb.block_dim], cb);
    }

    return ck;
}

CompressedValue quantize_value(const float* value, int head_dim, const Codebook& cb) {
    CompressedValue cv;
    cv.n_blocks = head_dim / cb.block_dim;
    cv.norm = 0.0f;

    for (int d = 0; d < head_dim; d++) {
        cv.norm += value[d] * value[d];
    }
    cv.norm = std::sqrt(cv.norm);

    for (int b = 0; b < cv.n_blocks; b++) {
        cv.indices[b] = (uint8_t)quantize_block(&value[b * cb.block_dim], cb);
    }

    return cv;
}

// ---- Decoding ----

void decode_key(const CompressedKey& ck, const Codebook& cb, float* out, int head_dim) {
    for (int b = 0; b < ck.n_blocks; b++) {
        const float* cw = cb.codeword(ck.indices[b]);
        std::memcpy(&out[b * cb.block_dim], cw, cb.block_dim * sizeof(float));
    }
}

void decode_value(const CompressedValue& cv, const Codebook& cb, float* out, int head_dim) {
    for (int b = 0; b < cv.n_blocks; b++) {
        const float* cw = cb.codeword(cv.indices[b]);
        std::memcpy(&out[b * cb.block_dim], cw, cb.block_dim * sizeof(float));
    }
}

// ---- Query preparation ----

PreparedQuery prepare_query(const float* query, int head_dim, const Codebook& cb) {
    PreparedQuery pq;
    pq.n_blocks = head_dim / cb.block_dim;

    // Compute query norm
    pq.query_norm = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        pq.query_norm += query[d] * query[d];
    }
    pq.query_norm = std::sqrt(pq.query_norm);

    // Quantize each query block
    for (int b = 0; b < pq.n_blocks; b++) {
        pq.query_indices[b] = (uint8_t)quantize_block(&query[b * cb.block_dim], cb);
    }

    return pq;
}

// ---- Gram row pre-fetch ----

PrefetchedGramRows prefetch_gram_rows(const PreparedQuery& pq, const GramTable& gram) {
    PrefetchedGramRows pr;
    pr.n = gram.n;
    pr.block_count = pq.n_blocks;
    pr.query_norm = pq.query_norm;
    pr.gram_rows.resize(pq.n_blocks * gram.n);

    for (int b = 0; b < pq.n_blocks; b++) {
        int qi = pq.query_indices[b];
        std::memcpy(&pr.gram_rows[b * gram.n], &gram.values[qi * gram.n], gram.n * sizeof(float));
    }

    return pr;
}

// ---- Prepared index ----

FullyPreparedIndex build_prepared_index(const std::vector<CompressedKey>& keys,
                                         int num_tokens, int head_dim,
                                         const Codebook& cb) {
    FullyPreparedIndex idx;
    idx.n_blocks = head_dim / cb.block_dim;
    idx.n = cb.n;
    idx.block_dim = cb.block_dim;
    idx.head_dim = head_dim;
    idx.num_tokens = num_tokens;

    idx.key_indices_flat.resize(num_tokens * idx.n_blocks);
    idx.key_norms.resize(num_tokens);

    for (int t = 0; t < num_tokens; t++) {
        for (int b = 0; b < idx.n_blocks; b++) {
            idx.key_indices_flat[t * idx.n_blocks + b] = keys[t].indices[b];
        }
        idx.key_norms[t] = keys[t].norm;
    }

    return idx;
}

// ---- Scoring (the hot loop) ----

std::vector<std::pair<int, float>> score_all_tokens(
    const FullyPreparedIndex& index,
    const PrefetchedGramRows& prefetched) {

    int num_tokens = index.num_tokens;
    int n_blocks = index.n_blocks;
    int n = prefetched.n;
    float q_norm = prefetched.query_norm;

    std::vector<std::pair<int, float>> scored(num_tokens);

    for (int t = 0; t < num_tokens; t++) {
        const uint8_t* k_indices = index.key_block(t);
        float k_norm = index.key_norms[t];

        float total = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            total += prefetched.gram_rows[b * n + k_indices[b]];
        }

        float score = total * q_norm * k_norm;
        scored[t] = {t, score};
    }

    return scored;
}

// ---- Full compressed attention ----

std::vector<AttentionHit> compressed_attention_topk(
    const FullyPreparedIndex& k_index,
    const std::vector<CompressedValue>& v_codes,
    const Codebook& cb,
    const float* query,
    int head_dim,
    int top_k) {

    // Prepare query
    PreparedQuery pq = prepare_query(query, head_dim, cb);

    // Pre-fetch Gram rows (not used directly here, but could be)
    // Instead, use inline scoring for clarity

    int num_tokens = k_index.num_tokens;
    int n_blocks = k_index.n_blocks;
    int n = cb.n;

    // Build Gram table for scoring
    GramTable gram = build_gram_table(cb);

    // Score all tokens using Gram table lookups
    std::vector<std::pair<int, float>> scored(num_tokens);
    for (int t = 0; t < num_tokens; t++) {
        const uint8_t* k_indices = k_index.key_block(t);
        float k_norm = k_index.key_norms[t];

        float total = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            int qi = pq.query_indices[b];
            int ki = k_indices[b];
            total += gram.at(qi, ki);
        }
        scored[t] = {t, total * pq.query_norm * k_norm};
    }

    // Sort descending, take top-k
    int selected = std::min(top_k, num_tokens);
    if (selected > 0 && selected < num_tokens) {
        std::nth_element(scored.begin(), scored.begin() + selected, scored.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }
    scored.resize(selected);
    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // Decode V vectors for selected tokens
    std::vector<AttentionHit> hits;
    hits.reserve(selected);
    for (const auto& [token_idx, score] : scored) {
        AttentionHit hit;
        hit.token_index = token_idx;
        hit.score = score;
        hit.value.resize(head_dim);
        decode_value(v_codes[token_idx], cb, hit.value.data(), head_dim);
        hits.push_back(std::move(hit));
    }

    return hits;
}

// ---- Batch multi-head compressed attention ----

std::vector<std::vector<AttentionHit>> compressed_attention_topk_batch(
    const FullyPreparedIndex& k_index,
    const std::vector<CompressedValue>& v_codes,
    const Codebook& cb,
    const float* const* queries,
    int num_heads,
    int head_dim,
    int top_k) {

    GramTable gram = build_gram_table(cb);

    // Prepare all queries
    std::vector<PreparedQuery> pqs(num_heads);
    for (int h = 0; h < num_heads; h++) {
        pqs[h] = prepare_query(queries[h], head_dim, cb);
    }

    int num_tokens = k_index.num_tokens;
    int n_blocks = k_index.n_blocks;
    int n = cb.n;

    // Score all heads for all tokens in one pass
    std::vector<std::vector<std::pair<int, float>>> all_scored(num_heads,
        std::vector<std::pair<int, float>>(num_tokens));

    for (int t = 0; t < num_tokens; t++) {
        const uint8_t* k_indices = k_index.key_block(t);
        float k_norm = k_index.key_norms[t];

        for (int h = 0; h < num_heads; h++) {
            float total = 0.0f;
            for (int b = 0; b < n_blocks; b++) {
                int qi = pqs[h].query_indices[b];
                int ki = k_indices[b];
                total += gram.at(qi, ki);
            }
            float score = total * pqs[h].query_norm * k_norm;
            all_scored[h][t] = {t, score};
        }
    }

    // Top-k selection per head + V decode
    std::vector<std::vector<AttentionHit>> results(num_heads);

    for (int h = 0; h < num_heads; h++) {
        auto& scored = all_scored[h];
        int selected = std::min(top_k, num_tokens);
        if (selected > 0 && selected < num_tokens) {
            std::nth_element(scored.begin(), scored.begin() + selected, scored.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
        }
        scored.resize(selected);
        std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        results[h].reserve(selected);
        for (const auto& [token_idx, score] : scored) {
            AttentionHit hit;
            hit.token_index = token_idx;
            hit.score = score;
            hit.value.resize(head_dim);
            decode_value(v_codes[token_idx], cb, hit.value.data(), head_dim);
            results[h].push_back(std::move(hit));
        }
    }

    return results;
}

// ---- Metrics ----

float compression_ratio(int head_dim, int block_dim, int codebook_size) {
    size_t f16 = f16_size_bytes(head_dim);
    size_t comp = compressed_size_bytes(head_dim, block_dim);
    return (float)f16 / (float)comp;
}

size_t compressed_size_bytes(int head_dim, int block_dim) {
    int n_blocks = head_dim / block_dim;
    return n_blocks * sizeof(uint8_t) + sizeof(float); // indices + norm
}

size_t f16_size_bytes(int head_dim) {
    return head_dim * sizeof(float) / 2; // F16 = 2 bytes per element
}

} // namespace ri_vq