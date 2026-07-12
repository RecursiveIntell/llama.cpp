// ri_vq_cuda.cu — CUDA scoring kernel for RI-VQ compressed attention
//
// Scores all tokens against a query using Gram table lookups in shared memory.
// This is the GPU equivalent of ri-vq.cpp's score_all_tokens + prefetch_gram_rows.
//
// GTX 1070 (Pascal, compute 6.1): 48KB shared memory per SM, 2048 threads per SM.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdint>

// Must match ri-vq.h types
using index_t = uint16_t;

// CUDA kernel: score all tokens against a query using prefetched Gram rows
__global__ void ri_vq_score_kernel(
    const index_t* __restrict__ key_indices,  // [n_tokens * n_blocks]
    const float*   __restrict__ key_norms,     // [n_tokens]
    const float*   __restrict__ gram_rows,     // [n_blocks * N] prefetched
    float          query_norm,
    float          scale,
    int            n_blocks,
    int            n_tokens,
    int            N,                           // codebook size
    float*         __restrict__ scores          // [n_tokens] output
) {
    // Load prefetched Gram rows into shared memory
    extern __shared__ float s_gram[];
    int tid = threadIdx.x;
    int row_size = n_blocks * N;
    for (int i = tid; i < row_size; i += blockDim.x) {
        s_gram[i] = gram_rows[i];
    }
    __syncthreads();

    // Each thread scores one or more tokens
    for (int token = blockIdx.x * blockDim.x + tid;
         token < n_tokens;
         token += gridDim.x * blockDim.x) {

        const index_t* k_block = &key_indices[(size_t)token * n_blocks];
        float k_norm = key_norms[token];

        float total = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            total += s_gram[(size_t)b * N + k_block[b]];
        }

        scores[token] = total * query_norm * k_norm * scale;
    }
}

// CUDA kernel: exact attention (dot product Q.K) for baseline comparison
__global__ void exact_attn_kernel(
    const float* __restrict__ keys,      // [n_tokens * head_dim]
    const float* __restrict__ key_norms,  // [n_tokens] (not used, but for compatibility)
    const float* __restrict__ query,      // [head_dim]
    float scale,
    int head_dim,
    int n_tokens,
    float* __restrict__ scores             // [n_tokens] output
) {
    int tid = threadIdx.x;
    for (int token = blockIdx.x * blockDim.x + tid;
         token < n_tokens;
         token += gridDim.x * blockDim.x) {

        const float* k = &keys[(size_t)token * head_dim];
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += query[d] * k[d];
        }
        scores[token] = dot * scale;
    }
}

// ---- Host-side helpers ----

static std::vector<float> make_random_data(int n, int dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data((size_t)n * dim);
    for (size_t i = 0; i < (size_t)n * dim; i++) data[i] = dist(rng);
    return data;
}

// Simple k-means++ training (same as ri-vq.cpp)
struct Codebook {
    std::vector<float> table;
    int n, block_dim;
    const float* codeword(int i) const { return &table[(size_t)i * block_dim]; }
    float* codeword(int i) { return &table[(size_t)i * block_dim]; }
};

Codebook train_codebook(const float* samples, int n_samples, int block_dim,
                        int codebook_size, int max_iter, uint64_t seed) {
    Codebook cb;
    cb.n = std::min(codebook_size, n_samples);
    cb.block_dim = block_dim;
    cb.table.resize((size_t)cb.n * block_dim);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, n_samples - 1);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    std::vector<float> min_dist_sq(n_samples, 1e30f);
    int first = dist(rng);
    std::memcpy(cb.codeword(0), &samples[first * block_dim], block_dim * sizeof(float));

    for (int i = 1; i < cb.n; i++) {
        const float* c = cb.codeword(i - 1);
        float total = 0;
        for (int s = 0; s < n_samples; s++) {
            float d = 0;
            for (int dd = 0; dd < block_dim; dd++) { float diff = samples[s*block_dim+dd] - c[dd]; d += diff*diff; }
            if (d < min_dist_sq[s]) min_dist_sq[s] = d;
            total += min_dist_sq[s];
        }
        if (total <= 0) { int idx = dist(rng); std::memcpy(cb.codeword(i), &samples[idx*block_dim], block_dim*sizeof(float)); }
        else { float r = uniform(rng)*total; float cs = 0; int ch = n_samples-1;
            for (int s = 0; s < n_samples; s++) { cs += min_dist_sq[s]; if (cs >= r) { ch = s; break; } }
            std::memcpy(cb.codeword(i), &samples[ch*block_dim], block_dim*sizeof(float)); }
    }

    std::vector<int> assign(n_samples, 0), counts(cb.n, 0);
    std::vector<float> nc((size_t)cb.n * block_dim, 0);
    for (int iter = 0; iter < max_iter; iter++) {
        bool changed = false;
        for (int s = 0; s < n_samples; s++) {
            float bd = 1e30f; int bi = 0;
            for (int c = 0; c < cb.n; c++) { float d = 0; for (int dd = 0; dd < block_dim; dd++) { float diff = samples[s*block_dim+dd]-cb.codeword(c)[dd]; d += diff*diff; } if (d < bd) { bd = d; bi = c; } }
            if (assign[s] != bi) { assign[s] = bi; changed = true; } }
        if (!changed && iter > 0) break;
        std::fill(counts.begin(), counts.end(), 0); std::fill(nc.begin(), nc.end(), 0);
        for (int s = 0; s < n_samples; s++) { int c = assign[s]; counts[c]++; for (int dd = 0; dd < block_dim; dd++) nc[c*block_dim+dd] += samples[s*block_dim+dd]; }
        for (int c = 0; c < cb.n; c++) { if (counts[c] > 0) { for (int dd = 0; dd < block_dim; dd++) nc[c*block_dim+dd] /= counts[c]; std::memcpy(cb.codeword(c), &nc[c*block_dim], block_dim*sizeof(float)); }
            else { int idx = dist(rng); std::memcpy(cb.codeword(c), &samples[idx*block_dim], block_dim*sizeof(float)); } }
    }
    return cb;
}

int main() {
    constexpr int HEAD_DIM = 128;
    constexpr int BLOCK_DIM = 4;
    constexpr int N = 256;
    constexpr int N_TOKENS = 2048;
    constexpr int N_BLOCKS = HEAD_DIM / BLOCK_DIM;
    constexpr float SCALE = 1.0f / std::sqrt((float)HEAD_DIM);

    printf("=== RI-VQ CUDA Benchmark (GTX 1070) ===\n");
    printf("head_dim=%d, block_dim=%d, N=%d, tokens=%d\n\n", HEAD_DIM, BLOCK_DIM, N, N_TOKENS);

    // Generate data
    auto key_data = make_random_data(N_TOKENS, HEAD_DIM, 67890);
    auto query_data = make_random_data(1, HEAD_DIM, 11111);
    auto train_data = make_random_data(16384, BLOCK_DIM, 12345);

    // Train codebook
    Codebook cb = train_codebook(train_data.data(), 16384, BLOCK_DIM, N, 500, 42);

    // Build Gram table
    std::vector<float> gram((size_t)N * N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dot = 0;
            for (int d = 0; d < BLOCK_DIM; d++) dot += cb.codeword(i)[d] * cb.codeword(j)[d];
            gram[(size_t)i * N + j] = dot;
        }

    // Quantize keys
    std::vector<index_t> key_indices((size_t)N_TOKENS * N_BLOCKS);
    std::vector<float> key_norms(N_TOKENS);
    for (int t = 0; t < N_TOKENS; t++) {
        float norm = 0;
        for (int d = 0; d < HEAD_DIM; d++) norm += key_data[(size_t)t*HEAD_DIM+d] * key_data[(size_t)t*HEAD_DIM+d];
        key_norms[t] = std::sqrt(norm);
        for (int b = 0; b < N_BLOCKS; b++) {
            float bd = 1e30f; int bi = 0;
            for (int c = 0; c < N; c++) { float d = 0; for (int dd = 0; dd < BLOCK_DIM; dd++) { float diff = key_data[(size_t)t*HEAD_DIM+b*BLOCK_DIM+dd]-cb.codeword(c)[dd]; d += diff*diff; } if (d < bd) { bd = d; bi = c; } }
            key_indices[(size_t)t * N_BLOCKS + b] = (index_t)bi;
        }
    }

    // Prepare query
    float q_norm = 0;
    for (int d = 0; d < HEAD_DIM; d++) q_norm += query_data[d] * query_data[d];
    q_norm = std::sqrt(q_norm);
    std::vector<index_t> q_indices(N_BLOCKS);
    for (int b = 0; b < N_BLOCKS; b++) {
        float bd = 1e30f; int bi = 0;
        for (int c = 0; c < N; c++) { float d = 0; for (int dd = 0; dd < BLOCK_DIM; dd++) { float diff = query_data[b*BLOCK_DIM+dd]-cb.codeword(c)[dd]; d += diff*diff; } if (d < bd) { bd = d; bi = c; } }
        q_indices[b] = (index_t)bi;
    }

    // Prefetch Gram rows
    std::vector<float> gram_rows((size_t)N_BLOCKS * N);
    for (int b = 0; b < N_BLOCKS; b++) {
        std::memcpy(&gram_rows[(size_t)b * N], &gram[(size_t)q_indices[b] * N], N * sizeof(float));
    }

    // ---- CPU baseline timing ----
    std::vector<float> cpu_scores(N_TOKENS);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 1000; rep++) {
        for (int t = 0; t < N_TOKENS; t++) {
            const index_t* kb = &key_indices[(size_t)t * N_BLOCKS];
            float total = 0;
            for (int b = 0; b < N_BLOCKS; b++) total += gram_rows[(size_t)b * N + kb[b]];
            cpu_scores[t] = total * q_norm * key_norms[t] * SCALE;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_us = std::chrono::duration<double, std::micro>(t1-t0).count() / 1000.0;
    printf("CPU compressed scoring: %.2f us/call (%.2f ns/token)\n", cpu_us, cpu_us*1000/N_TOKENS);

    // ---- CPU exact baseline ----
    std::vector<float> exact_scores(N_TOKENS);
    t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 1000; rep++) {
        for (int t = 0; t < N_TOKENS; t++) {
            float dot = 0;
            for (int d = 0; d < HEAD_DIM; d++) dot += query_data[d] * key_data[(size_t)t*HEAD_DIM+d];
            exact_scores[t] = dot * SCALE;
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_exact_us = std::chrono::duration<double, std::micro>(t1-t0).count() / 1000.0;
    printf("CPU exact attention:   %.2f us/call (%.2f ns/token)\n", cpu_exact_us, cpu_exact_us*1000/N_TOKENS);

    // ---- GPU setup ----
    float *d_key_indices, *d_key_norms, *d_gram_rows, *d_scores;
    float *d_query, *d_keys;
    cudaMalloc(&d_key_indices, key_indices.size() * sizeof(index_t));
    cudaMalloc(&d_key_norms, N_TOKENS * sizeof(float));
    cudaMalloc(&d_gram_rows, gram_rows.size() * sizeof(float));
    cudaMalloc(&d_scores, N_TOKENS * sizeof(float));
    cudaMalloc(&d_query, HEAD_DIM * sizeof(float));
    cudaMalloc(&d_keys, key_data.size() * sizeof(float));

    cudaMemcpy(d_key_indices, key_indices.data(), key_indices.size()*sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key_norms, key_norms.data(), N_TOKENS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gram_rows, gram_rows.data(), gram_rows.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query_data.data(), HEAD_DIM*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_keys, key_data.data(), key_data.size()*sizeof(float), cudaMemcpyHostToDevice);

    // ---- GPU compressed scoring ----
    int block_size = 256;
    int grid_size = (N_TOKENS + block_size - 1) / block_size;
    size_t shared_bytes = N_BLOCKS * N * sizeof(float);

    std::vector<float> gpu_scores(N_TOKENS);

    // Warmup
    for (int i = 0; i < 5; i++)
        ri_vq_score_kernel<<<grid_size, block_size, shared_bytes>>>(
            (index_t*)d_key_indices, d_key_norms, d_gram_rows, q_norm, SCALE,
            N_BLOCKS, N_TOKENS, N, d_scores);
    cudaDeviceSynchronize();

    // Time GPU compressed
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int rep = 0; rep < 1000; rep++)
        ri_vq_score_kernel<<<grid_size, block_size, shared_bytes>>>(
            (index_t*)d_key_indices, d_key_norms, d_gram_rows, q_norm, SCALE,
            N_BLOCKS, N_TOKENS, N, d_scores);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    double gpu_us = gpu_ms / 1000.0 * 1000.0; // ms -> us/call for 1000 reps
    printf("GPU compressed scoring: %.2f us/call (%.2f ns/token)\n", gpu_us, gpu_us*1000/N_TOKENS);

    cudaMemcpy(gpu_scores.data(), d_scores, N_TOKENS*sizeof(float), cudaMemcpyDeviceToHost);

    // ---- GPU exact attention ----
    cudaEventRecord(start);
    for (int rep = 0; rep < 1000; rep++)
        exact_attn_kernel<<<grid_size, block_size>>>(
            d_keys, d_key_norms, d_query, SCALE, HEAD_DIM, N_TOKENS, d_scores);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ms, start, stop);
    double gpu_exact_us = gpu_ms / 1000.0 * 1000.0;
    printf("GPU exact attention:    %.2f us/call (%.2f ns/token)\n", gpu_exact_us, gpu_exact_us*1000/N_TOKENS);

    // ---- Verify GPU matches CPU ----
    float max_err = 0;
    for (int t = 0; t < N_TOKENS; t++) {
        float err = std::abs(gpu_scores[t] - cpu_scores[t]);
        if (err > max_err) max_err = err;
    }
    printf("\nGPU vs CPU max error: %.6f\n", max_err);

    // ---- Speedup summary ----
    printf("\n=== Speedup Summary ===\n");
    printf("GPU compressed vs CPU compressed: %.2fx\n", cpu_us / gpu_us);
    printf("GPU compressed vs CPU exact:     %.2fx\n", cpu_exact_us / gpu_us);
    printf("GPU exact vs CPU exact:           %.2fx\n", cpu_exact_us / gpu_exact_us);
    printf("CPU compressed vs CPU exact:      %.2fx\n", cpu_exact_us / cpu_us);

    // ---- Scale test: 32K tokens ----
    printf("\n=== Scale test: 32K tokens ===\n");
    constexpr int N_TOKENS_LARGE = 32768;
    auto large_keys = make_random_data(N_TOKENS_LARGE, HEAD_DIM, 99999);
    std::vector<index_t> large_key_indices((size_t)N_TOKENS_LARGE * N_BLOCKS);
    std::vector<float> large_key_norms(N_TOKENS_LARGE);
    for (int t = 0; t < N_TOKENS_LARGE; t++) {
        float norm = 0;
        for (int d = 0; d < HEAD_DIM; d++) norm += large_keys[(size_t)t*HEAD_DIM+d]*large_keys[(size_t)t*HEAD_DIM+d];
        large_key_norms[t] = std::sqrt(norm);
        for (int b = 0; b < N_BLOCKS; b++) {
            float bd = 1e30f; int bi = 0;
            for (int c = 0; c < N; c++) { float d = 0; for (int dd = 0; dd < BLOCK_DIM; dd++) { float diff = large_keys[(size_t)t*HEAD_DIM+b*BLOCK_DIM+dd]-cb.codeword(c)[dd]; d += diff*diff; } if (d < bd) { bd = d; bi = c; } }
            large_key_indices[(size_t)t * N_BLOCKS + b] = (index_t)bi;
        }
    }

    // CPU 32K
    t0 = std::chrono::high_resolution_clock::now();
    for (int rep = 0; rep < 100; rep++) {
        for (int t = 0; t < N_TOKENS_LARGE; t++) {
            const index_t* kb = &large_key_indices[(size_t)t * N_BLOCKS];
            float total = 0;
            for (int b = 0; b < N_BLOCKS; b++) total += gram_rows[(size_t)b * N + kb[b]];
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_large_us = std::chrono::duration<double, std::micro>(t1-t0).count() / 100.0;

    // GPU 32K
    float *d_large_indices, *d_large_norms, *d_large_scores;
    cudaMalloc(&d_large_indices, large_key_indices.size()*sizeof(index_t));
    cudaMalloc(&d_large_norms, N_TOKENS_LARGE*sizeof(float));
    cudaMalloc(&d_large_scores, N_TOKENS_LARGE*sizeof(float));
    cudaMemcpy(d_large_indices, large_key_indices.data(), large_key_indices.size()*sizeof(index_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_large_norms, large_key_norms.data(), N_TOKENS_LARGE*sizeof(float), cudaMemcpyHostToDevice);

    int grid_large = (N_TOKENS_LARGE + block_size - 1) / block_size;
    for (int i = 0; i < 5; i++)
        ri_vq_score_kernel<<<grid_large, block_size, shared_bytes>>>(
            (index_t*)d_large_indices, d_large_norms, d_gram_rows, q_norm, SCALE,
            N_BLOCKS, N_TOKENS_LARGE, N, d_large_scores);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int rep = 0; rep < 100; rep++)
        ri_vq_score_kernel<<<grid_large, block_size, shared_bytes>>>(
            (index_t*)d_large_indices, d_large_norms, d_gram_rows, q_norm, SCALE,
            N_BLOCKS, N_TOKENS_LARGE, N, d_large_scores);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_ms, start, stop);
    double gpu_large_us = gpu_ms / 100.0;

    printf("CPU compressed 32K: %.2f us/call\n", cpu_large_us);
    printf("GPU compressed 32K: %.2f us/call\n", gpu_large_us);
    printf("GPU vs CPU 32K:     %.2fx\n", cpu_large_us / gpu_large_us);

    // Cleanup
    cudaFree(d_key_indices); cudaFree(d_key_norms); cudaFree(d_gram_rows);
    cudaFree(d_scores); cudaFree(d_query); cudaFree(d_keys);
    cudaFree(d_large_indices); cudaFree(d_large_norms); cudaFree(d_large_scores);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("\n=== done ===\n");
    return 0;
}