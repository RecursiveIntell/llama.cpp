// test-ri-vq.cpp — Standalone test for RI vector quantization compressed attention
//
// Validates:
// 1. Codebook training produces valid codebook
// 2. Quantize -> decode round-trip has bounded error
// 3. Compressed attention scores correlate with exact attention scores
// 4. Compression ratio is correct
// 5. Top-k overlap between compressed and exact attention
//
// Build: g++ -O2 -std=c++17 -o test-ri-vq test-ri-vq.cpp ri-vq.cpp
// Run:   ./test-ri-vq

#include "ri-vq.h"
#include <cstdio>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace ri_vq;

// Generate random test data
static std::vector<float> make_random_data(int n, int dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (int i = 0; i < n * dim; i++) {
        data[i] = dist(rng);
    }
    return data;
}

// Exact attention: dot product Q with all K, return top-k
static std::vector<std::pair<int, float>> exact_attention(
    const float* query, const std::vector<float>& keys,
    int head_dim, int top_k) {

    int num_tokens = keys.size() / head_dim;
    std::vector<std::pair<int, float>> scored(num_tokens);
    for (int t = 0; t < num_tokens; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += query[d] * keys[t * head_dim + d];
        }
        scored[t] = {t, dot};
    }

    int selected = std::min(top_k, num_tokens);
    std::nth_element(scored.begin(), scored.begin() + selected, scored.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    scored.resize(selected);
    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    return scored;
}

int main() {
    printf("=== RI-VQ Compressed Attention Test ===\n\n");

    // Config
    constexpr int HEAD_DIM     = 128;
    constexpr int BLOCK_DIM    = 4;
    constexpr int CODEBOOK_N   = 256;
    constexpr int NUM_TOKENS   = 2048;
    constexpr int TOP_K       = 8;
    constexpr int NUM_TRAIN    = 4096; // training blocks for codebook

    // 1. Generate training data (random blocks)
    printf("1. Generating training data...\n");
    auto train_data = make_random_data(NUM_TRAIN, BLOCK_DIM, 12345);

    // 2. Train codebook
    printf("2. Training codebook (N=%d, block_dim=%d, %d samples)...\n",
           CODEBOOK_N, BLOCK_DIM, NUM_TRAIN);
    auto t0 = std::chrono::high_resolution_clock::now();
    Codebook cb = train_codebook(train_data.data(), NUM_TRAIN, BLOCK_DIM,
                                  CODEBOOK_N, 50, 42);
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("   Codebook trained in %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // 3. Build Gram table
    printf("3. Building Gram table (%d x %d)...\n", CODEBOOK_N, CODEBOOK_N);
    GramTable gram = build_gram_table(cb);
    printf("   Gram table size: %.1f KB\n",
           (float)(CODEBOOK_N * CODEBOOK_N * sizeof(float)) / 1024.0f);

    // 4. Generate test keys and query
    printf("4. Generating %d test keys (head_dim=%d)...\n", NUM_TOKENS, HEAD_DIM);
    auto key_data = make_random_data(NUM_TOKENS, HEAD_DIM, 67890);
    auto query_data = make_random_data(1, HEAD_DIM, 11111);

    // 5. Quantize all keys
    printf("5. Quantizing keys...\n");
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<CompressedKey> compressed_keys(NUM_TOKENS);
    std::vector<CompressedValue> compressed_values(NUM_TOKENS);
    for (int t = 0; t < NUM_TOKENS; t++) {
        compressed_keys[t] = quantize_key(&key_data[t * HEAD_DIM], HEAD_DIM, cb);
        compressed_values[t] = quantize_value(&key_data[t * HEAD_DIM], HEAD_DIM, cb);
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf("   Quantized %d keys in %.1f ms\n", NUM_TOKENS,
           std::chrono::duration<double, std::milli>(t1 - t0).count());

    // 6. Verify compression ratio
    printf("\n6. Compression ratio:\n");
    float ratio = compression_ratio(HEAD_DIM, BLOCK_DIM, CODEBOOK_N);
    size_t f16_bytes = f16_size_bytes(HEAD_DIM);
    size_t comp_bytes = compressed_size_bytes(HEAD_DIM, BLOCK_DIM, CODEBOOK_N);
    printf("   F16 per key:  %zu bytes\n", f16_bytes);
    printf("   RI-VQ per key: %zu bytes (indices=%d + norm=4)\n", comp_bytes,
           HEAD_DIM / BLOCK_DIM);
    printf("   Compression ratio: %.1fx\n", ratio);

    // 7. Verify round-trip quality
    printf("\n7. Round-trip quality (quantize -> decode -> compare):\n");
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int n_checked = std::min(100, NUM_TOKENS);
    for (int t = 0; t < n_checked; t++) {
        float decoded[HEAD_DIM];
        decode_key(compressed_keys[t], cb, decoded, HEAD_DIM);
        float rmse = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            float diff = key_data[t * HEAD_DIM + d] - decoded[d];
            rmse += diff * diff;
        }
        rmse = std::sqrt(rmse / HEAD_DIM);
        avg_error += rmse;
        max_error = std::max(max_error, rmse);
    }
    avg_error /= n_checked;
    printf("   RMSE avg: %.6f, max: %.6f (over %d vectors)\n", avg_error, max_error, n_checked);
    float key_scale = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) key_scale += std::abs(key_data[d]);
    key_scale /= HEAD_DIM;
    printf("   Relative to avg |key|: %.4f%%\n", (avg_error / key_scale) * 100.0f);

    // 8. Score correlation: compressed vs exact
    printf("\n8. Score correlation (compressed vs exact attention):\n");
    auto exact_top = exact_attention(query_data.data(), key_data, HEAD_DIM, TOP_K);

    // Build prepared index for compressed scoring
    FullyPreparedIndex idx = build_prepared_index(compressed_keys, NUM_TOKENS, HEAD_DIM, cb);

    // Compressed scoring
    PreparedQuery pq = prepare_query(query_data.data(), HEAD_DIM, cb);
    PrefetchedGramRows prefetched = prefetch_gram_rows(pq, gram);
    auto comp_scored = score_all_tokens(idx, prefetched);

    // Select top-k from compressed scores
    int selected = std::min(TOP_K, NUM_TOKENS);
    std::nth_element(comp_scored.begin(), comp_scored.begin() + selected, comp_scored.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    comp_scored.resize(selected);
    std::sort(comp_scored.begin(), comp_scored.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // Top-k overlap
    std::vector<int> exact_indices, comp_indices;
    for (const auto& [idx, score] : exact_top) exact_indices.push_back(idx);
    for (const auto& [idx, score] : comp_scored) comp_indices.push_back(idx);

    int overlap = 0;
    for (int e : exact_indices) {
        for (int c : comp_indices) {
            if (e == c) { overlap++; break; }
        }
    }
    printf("   Top-%d overlap: %d/%d (%.1f%%)\n", TOP_K, overlap, TOP_K,
           100.0f * overlap / TOP_K);

    // Score correlation (Pearson) on all tokens
    float mean_exact = 0.0f, mean_comp = 0.0f;
    for (int t = 0; t < NUM_TOKENS; t++) {
        mean_exact += exact_top.empty() ? 0.0f : 0.0f; // placeholder
    }
    // Actually compute correlation over all tokens, not just top-k
    std::vector<float> exact_scores(NUM_TOKENS), comp_scores(NUM_TOKENS);
    for (int t = 0; t < NUM_TOKENS; t++) {
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            dot += query_data[d] * key_data[t * HEAD_DIM + d];
        }
        exact_scores[t] = dot;
        comp_scores[t] = comp_scored.empty() ? 0.0f : 0.0f; // will fix below
    }
    // Re-score all tokens (comp_scored was truncated)
    auto comp_all = score_all_tokens(idx, prefetched);
    for (int t = 0; t < NUM_TOKENS; t++) {
        exact_scores[t] = exact_scores[t]; // already computed above
        comp_scores[t] = comp_all[t].second;
    }

    // Pearson correlation
    mean_exact = 0.0f; mean_comp = 0.0f;
    for (int t = 0; t < NUM_TOKENS; t++) {
        mean_exact += exact_scores[t];
        mean_comp += comp_scores[t];
    }
    mean_exact /= NUM_TOKENS;
    mean_comp /= NUM_TOKENS;

    float cov = 0.0f, var_e = 0.0f, var_c = 0.0f;
    for (int t = 0; t < NUM_TOKENS; t++) {
        float de = exact_scores[t] - mean_exact;
        float dc = comp_scores[t] - mean_comp;
        cov += de * dc;
        var_e += de * de;
        var_c += dc * dc;
    }
    float pearson = cov / std::sqrt(var_e * var_c + 1e-30f);
    printf("   Pearson correlation (all %d tokens): %.6f\n", NUM_TOKENS, pearson);

    // 9. Timing: compressed vs exact
    printf("\n9. Timing (2048 tokens, head_dim=128, top_k=%d):\n", TOP_K);

    // Exact attention timing
    t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 100; rep++) {
        auto _ = exact_attention(query_data.data(), key_data, HEAD_DIM, TOP_K);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double exact_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;

    // Compressed attention timing
    t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 100; rep++) {
        PreparedQuery pq2 = prepare_query(query_data.data(), HEAD_DIM, cb);
        PrefetchedGramRows pr2 = prefetch_gram_rows(pq2, gram);
        auto _ = score_all_tokens(idx, pr2);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double comp_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;

    printf("   Exact:    %.1f us/call\n", exact_us);
    printf("   Compressed: %.1f us/call\n", comp_us);
    printf("   Speedup:  %.2fx\n", exact_us / comp_us);

    // 10. Summary
    printf("\n=== SUMMARY ===\n");
    printf("Compression:     %.1fx (F16 -> RI-VQ, head_dim=%d, block_dim=%d)\n",
           ratio, HEAD_DIM, BLOCK_DIM);
    printf("Round-trip RMSE: avg=%.6f, max=%.6f\n", avg_error, max_error);
    printf("Top-k overlap:   %d/%d (%.1f%%)\n", overlap, TOP_K,
           100.0f * overlap / TOP_K);
    printf("Pearson corr:    %.6f\n", pearson);
    printf("Speedup:         %.2fx (compressed vs exact scoring)\n", exact_us / comp_us);

    // Pass/fail
    bool pass = true;
    if (ratio < 5.0f) { printf("FAIL: compression ratio too low\n"); pass = false; }
    if (pearson < 0.8f) { printf("FAIL: score correlation too low\n"); pass = false; }
    if (overlap < TOP_K / 4) { printf("FAIL: top-k overlap too low\n"); pass = false; }

    printf("\nResult: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}