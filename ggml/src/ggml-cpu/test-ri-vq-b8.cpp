#include "ri-vq.h"
#include <cstdio>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace ri_vq;

static std::vector<float> make_random_data(int n, int dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (int i = 0; i < n * dim; i++) data[i] = dist(rng);
    return data;
}

static std::vector<std::pair<int, float>> exact_attn(
    const float* query, const std::vector<float>& keys, int head_dim, int top_k) {
    int num_tokens = keys.size() / head_dim;
    std::vector<std::pair<int, float>> scored(num_tokens);
    for (int t = 0; t < num_tokens; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += query[d] * keys[t * head_dim + d];
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
    constexpr int HEAD_DIM = 128, NUM_TOKENS = 2048, TOP_K = 8;
    int block_dims[] = {4, 8, 16};
    int codebook_sizes[] = {256};
    
    printf("=== RI-VQ Parameter Sweep (head_dim=%d, tokens=%d) ===\n\n", HEAD_DIM, NUM_TOKENS);
    printf("%-6s %-6s %-8s %-10s %-10s %-12s %-10s %-10s\n",
           "blk", "N", "ratio", "rmse_avg", "rmse_max", "pearson", "overlap", "speedup");
    printf("----------------------------------------------------------------------\n");

    auto key_data = make_random_data(NUM_TOKENS, HEAD_DIM, 67890);
    auto query_data = make_random_data(1, HEAD_DIM, 11111);

    for (int bd : block_dims) {
        for (int cn : codebook_sizes) {
            int n_blocks = HEAD_DIM / bd;
            int train_n = std::max(16384, cn * 16);

            auto train_data = make_random_data(train_n, bd, 12345);
            Codebook cb = train_codebook(train_data.data(), train_n, bd, cn, 500, 42);
            GramTable gram = build_gram_table(cb);

            std::vector<CompressedKey> keys(NUM_TOKENS);
            for (int t = 0; t < NUM_TOKENS; t++)
                keys[t] = quantize_key(&key_data[t * HEAD_DIM], HEAD_DIM, cb);

            FullyPreparedIndex idx = build_prepared_index(keys, NUM_TOKENS, HEAD_DIM, cb);

            // Round-trip RMSE
            float rmse_avg = 0, rmse_max = 0;
            int n_check = std::min(100, NUM_TOKENS);
            for (int t = 0; t < n_check; t++) {
                float dec[HEAD_DIM];
                decode_key(keys[t], cb, dec, HEAD_DIM);
                float rmse = 0;
                for (int d = 0; d < HEAD_DIM; d++) {
                    float diff = key_data[t * HEAD_DIM + d] - dec[d];
                    rmse += diff * diff;
                }
                rmse = std::sqrt(rmse / HEAD_DIM);
                rmse_avg += rmse;
                rmse_max = std::max(rmse_max, rmse);
            }
            rmse_avg /= n_check;

            // Pearson correlation
            PreparedQuery pq = prepare_query(query_data.data(), HEAD_DIM, cb);
            PrefetchedGramRows pr = prefetch_gram_rows(pq, gram);
            auto comp_all = score_all_tokens(idx, pr);

            std::vector<float> es(NUM_TOKENS), cs(NUM_TOKENS);
            for (int t = 0; t < NUM_TOKENS; t++) {
                float dot = 0;
                for (int d = 0; d < HEAD_DIM; d++)
                    dot += query_data[d] * key_data[t * HEAD_DIM + d];
                es[t] = dot;
                cs[t] = comp_all[t].second;
            }
            float me = 0, mc = 0;
            for (int t = 0; t < NUM_TOKENS; t++) { me += es[t]; mc += cs[t]; }
            me /= NUM_TOKENS; mc /= NUM_TOKENS;
            float cov = 0, ve = 0, vc = 0;
            for (int t = 0; t < NUM_TOKENS; t++) {
                float de = es[t] - me, dc = cs[t] - mc;
                cov += de * dc; ve += de * de; vc += dc * dc;
            }
            float pearson = cov / std::sqrt(ve * vc + 1e-30f);

            // Top-k overlap
            auto exact_top = exact_attn(query_data.data(), key_data, HEAD_DIM, TOP_K);
            int selected = std::min(TOP_K, NUM_TOKENS);
            std::nth_element(comp_all.begin(), comp_all.begin() + selected, comp_all.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            comp_all.resize(selected);
            std::sort(comp_all.begin(), comp_all.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            int overlap = 0;
            for (auto& [ei, _] : exact_top)
                for (auto& [ci, __] : comp_all)
                    if (ei == ci) { overlap++; break; }

            // Timing
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < 100; r++) {
                auto _ = exact_attn(query_data.data(), key_data, HEAD_DIM, TOP_K);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double exact_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;

            t0 = std::chrono::high_resolution_clock::now();
            for (int r = 0; r < 100; r++) {
                PreparedQuery pq2 = prepare_query(query_data.data(), HEAD_DIM, cb);
                PrefetchedGramRows pr2 = prefetch_gram_rows(pq2, gram);
                auto _ = score_all_tokens(idx, pr2);
            }
            t1 = std::chrono::high_resolution_clock::now();
            double comp_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;

            float ratio = compression_ratio(HEAD_DIM, bd, cn);
            printf("%-6d %-6d %-8.1f %-10.6f %-10.6f %-12.6f %-10d %-10.2f\n",
                   bd, cn, ratio, rmse_avg, rmse_max, pearson, overlap, exact_us / comp_us);
        }
    }
    return 0;
}
