// test-ri-vq-b8.cpp — RI-VQ parameter sweep with N=256 and N=1024
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
    std::vector<float> data((size_t)n * dim);
    for (size_t i = 0; i < (size_t)n * dim; i++) data[i] = dist(rng);
    return data;
}

static std::vector<std::pair<int, float>> exact_attn(
    const float* query, const std::vector<float>& keys, int head_dim, int top_k) {
    int num_tokens = (int)(keys.size() / head_dim);
    std::vector<std::pair<int, float>> scored(num_tokens);
    for (int t = 0; t < num_tokens; t++) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) dot += query[d] * keys[(size_t)t * head_dim + d];
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

    printf("=== RI-VQ Parameter Sweep (head_dim=%d, tokens=%d) ===\n\n", HEAD_DIM, NUM_TOKENS);
    printf("%-5s %-6s %-8s %-10s %-10s %-12s %-10s %-10s\n",
           "blk", "N", "ratio", "rmse_avg", "rmse_max", "pearson", "overlap", "speedup");
    printf("----------------------------------------------------------------------\n");

    auto key_data = make_random_data(NUM_TOKENS, HEAD_DIM, 67890);
    auto query_data = make_random_data(1, HEAD_DIM, 11111);

    struct TestConfig { int bd; int cn; };
    TestConfig configs[] = {
        {4, 256}, {8, 256}, {16, 256},
        {4, 1024}, {8, 1024}, {16, 1024},
    };

    for (auto [bd, cn] : configs) {
        int n_blocks = HEAD_DIM / bd;
        int train_n = std::max(16384, cn * 16);

        auto train_data = make_random_data(train_n, bd, 12345);
        Codebook cb = train_codebook(train_data.data(), train_n, bd, cn, 500, 42);
        GramTable gram = build_gram_table(cb);

        std::vector<CompressedKey> keys(NUM_TOKENS);
        for (int t = 0; t < NUM_TOKENS; t++)
            keys[t] = quantize_key(&key_data[(size_t)t * HEAD_DIM], HEAD_DIM, cb);

        FullyPreparedIndex idx = build_prepared_index(keys, NUM_TOKENS, HEAD_DIM, cb);

        // Round-trip RMSE
        float rmse_sum = 0, rmse_max = 0;
        int n_check = std::min(100, NUM_TOKENS);
        for (int t = 0; t < n_check; t++) {
            std::vector<float> dec(HEAD_DIM);
            decode_key(keys[t], cb, dec.data(), HEAD_DIM);
            float rmse = 0;
            for (int d = 0; d < HEAD_DIM; d++) {
                float diff = key_data[(size_t)t * HEAD_DIM + d] - dec[d];
                rmse += diff * diff;
            }
            rmse = std::sqrt(rmse / HEAD_DIM);
            rmse_sum += rmse;
            rmse_max = std::max(rmse_max, rmse);
        }
        float rmse_avg = rmse_sum / n_check;

        // Score correlation
        auto exact_top = exact_attn(query_data.data(), key_data, HEAD_DIM, TOP_K);
        PreparedQuery pq = prepare_query(query_data.data(), HEAD_DIM, cb);
        PrefetchedGramRows pr = prefetch_gram_rows(pq, gram);
        auto scored = score_all_tokens(idx, pr);

        // Pearson correlation
        float mean_e = 0, mean_c = 0;
        for (int t = 0; t < NUM_TOKENS; t++) {
            float e = exact_attn(query_data.data(), {key_data}, HEAD_DIM, NUM_TOKENS)[t].second;
            mean_e += e;
            mean_c += scored[t].second;
        }
        mean_e /= NUM_TOKENS; mean_c /= NUM_TOKENS;
        float cov = 0, var_e = 0, var_c = 0;
        for (int t = 0; t < NUM_TOKENS; t++) {
            float e = 0;
            for (int d = 0; d < HEAD_DIM; d++) e += query_data[d] * key_data[(size_t)t * HEAD_DIM + d];
            float de = e - mean_e, dc = scored[t].second - mean_c;
            cov += de * dc; var_e += de * de; var_c += dc * dc;
        }
        float pearson = (var_e > 0 && var_c > 0) ? cov / std::sqrt(var_e * var_c) : 0;

        // Top-k overlap
        std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        int overlap = 0;
        for (int i = 0; i < std::min(TOP_K, NUM_TOKENS); i++) {
            for (int j = 0; j < std::min(TOP_K, NUM_TOKENS); j++) {
                if (exact_top[i].first == scored[j].first) { overlap++; break; }
            }
        }

        // Timing
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 100; rep++) {
            PreparedQuery pq2 = prepare_query(query_data.data(), HEAD_DIM, cb);
            PrefetchedGramRows pr2 = prefetch_gram_rows(pq2, gram);
            auto s = score_all_tokens(idx, pr2);
            (void)s;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double comp_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;

        t0 = std::chrono::high_resolution_clock::now();
        for (int rep = 0; rep < 100; rep++) {
            auto e = exact_attn(query_data.data(), key_data, HEAD_DIM, TOP_K);
            (void)e;
        }
        t1 = std::chrono::high_resolution_clock::now();
        double exact_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / 100.0;

        float speedup = (float)(exact_us / comp_us);
        float ratio = compression_ratio(HEAD_DIM, bd, cn);

        printf("%-5d %-6d %-8.1f %-10.6f %-10.6f %-12.6f %-10d %-10.2f\n",
               bd, cn, ratio, rmse_avg, rmse_max, pearson, overlap, speedup);
    }

    printf("\n=== done ===\n");
    return 0;
}