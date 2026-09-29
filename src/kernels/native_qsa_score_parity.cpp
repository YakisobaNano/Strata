#include "strata/kernels/native_qsa_score.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "native_qsa_score_parity: %s: %s\n", where, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T> T* device_copy(const std::vector<T>& src, cudaStream_t stream) {
    T* p = nullptr;
    ck(cudaMalloc(&p, src.size() * sizeof(T)), "cudaMalloc");
    ck(cudaMemcpyAsync(p, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream), "upload");
    return p;
}

float reference_score(const float* pooled, const float* query, const float* bias, int row, int n) {
    float h[4]{};
    for (int head = 0; head < 4; ++head) {
        double sum = 0.0;
        for (int k = 0; k < 128; ++k)
            sum += (double) pooled[(size_t) row * 128 + k] * query[head * 128 + k];
        h[head] = std::fmax((float) sum, 0.0f);
    }
    float score = ((h[0] + h[1]) + h[2]) + h[3];
    score += bias[row];
    if (row == n / 4 && n % 4) score += 1e9f;
    return score + 0.0f;
}

void selection_check(cudaStream_t stream) {
    constexpr int nq = 3, max_blocks = 1025, cap = 2051;
    const auto shapes = strata::kernels::qsa_real_shapes();
    const int lengths[nq] = {16, 2051, 4097};
    std::vector<float> pooled(max_blocks * 128), dead(128), query(nq * 4 * 128);
    for (size_t i = 0; i < pooled.size(); ++i) pooled[i] = std::sin(float(i) * .031f);
    for (int d = 0; d < 128; ++d) dead[d] = std::cos(d * .07f);
    for (size_t i = 0; i < query.size(); ++i) query[i] = std::cos(float(i) * .019f);
    std::vector<int32_t> steps(nq * strata::kernels::kStepCount), ids(nq * cap, -1);
    for (int q = 0; q < nq; ++q) strata::kernels::qsa_step_fill(steps.data() + q * strata::kernels::kStepCount, lengths[q] - 1, shapes);
    std::vector<float> scores(nq * max_blocks, -1234);
    auto* dp = device_copy(pooled, stream); auto* dd = device_copy(dead, stream);
    auto* dq = device_copy(query, stream); auto* ds = device_copy(steps, stream);
    auto* dout = device_copy(scores, stream); auto* di = device_copy(ids, stream);
    int device = 0; cudaDeviceProp prop{};
    ck(cudaGetDevice(&device), "device"); ck(cudaGetDeviceProperties(&prop, device), "properties");
    if (prop.major < 8) {
        if (strata::kernels::qsa_block_scores_tc(dp, dd, dq, ds, nq, max_blocks, shapes, dout, stream, max_blocks)) {
            std::fprintf(stderr, "TF32 selection unexpectedly accepted sm_%d%d\n", prop.major, prop.minor); std::exit(1);
        }
        ck(cudaMemcpyAsync(scores.data(), dout, scores.size()*4, cudaMemcpyDeviceToHost, stream), "guard read");
        ck(cudaStreamSynchronize(stream), "guard sync");
        for (float x : scores) if (x != -1234) { std::fprintf(stderr,"TC refusal wrote output\n"); std::exit(1); }
    }
    strata::kernels::qsa_block_scores(dp, dd, dq, ds, nq, max_blocks, shapes, dout, stream);
    strata::kernels::qsa_block_topk(dout, ds, nq, max_blocks, cap, shapes, di, stream);
    ck(cudaMemcpyAsync(scores.data(), dout, scores.size()*4, cudaMemcpyDeviceToHost, stream), "scores read");
    ck(cudaMemcpyAsync(ids.data(), di, ids.size()*4, cudaMemcpyDeviceToHost, stream), "ids read");
    ck(cudaStreamSynchronize(stream), "selection sync");
    for (int q = 0; q < nq; ++q) {
        const int n = lengths[q], full = n / 4;
        for (int row = 0; row <= full; ++row) {
            const float* key = row == full ? dead.data() : pooled.data()+row*128;
            double ref = 0;
            for (int h = 0; h < 4; ++h) {
                double dot = 0;
                for (int d = 0; d < 128; ++d) dot += double(key[d])*query[(q*4+h)*128+d];
                ref += std::max(dot, 0.0);
            }
            if (row == full && n%4) ref += 1e9;
            const float got = scores[q*max_blocks+row];
            if (!std::isfinite(got) || std::abs(double(got)-ref)>2e-5*std::max(1.0,std::abs(ref))) {
                std::fprintf(stderr,"portable selection score mismatch q=%d row=%d\n",q,row);std::exit(1);
            }
        }
        std::vector<int> reference(n);
        for (int i=0;i<n;++i) reference[i]=i;
        std::sort(reference.begin(),reference.end(),[&](int a,int b) {
            float sa=scores[q*max_blocks+a/4], sb=scores[q*max_blocks+b/4];
            return sa==sb ? a<b : sa>sb;
        });
        const int width=steps[q*strata::kernels::kStepCount+strata::kernels::kStepWidth];
        reference.resize(width);std::sort(reference.begin(),reference.end());
        for (int i=0;i<width;++i) if(ids[q*cap+i]!=reference[i]) {
            std::fprintf(stderr,"portable top-k mismatch q=%d i=%d\n",q,i);std::exit(1);
        }
    }
    cudaFree(di);cudaFree(dout);cudaFree(ds);cudaFree(dq);cudaFree(dd);cudaFree(dp);
    std::printf("qsa_select portable: FP64 scores + CPU top-k PASS (16/2051/4097 cells); pre-sm80 TF32 refusal checked\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) != "--selftest") {
        std::fprintf(stderr, "usage: native_qsa_score_parity [--selftest]\n");
        return 2;
    }
    constexpr int max_cells = 2055, max_blocks = max_cells / 4 + 1;
    const auto shapes = strata::kernels::qsa_real_shapes();
    std::vector<float> pooled((size_t) max_blocks * 128), query(4 * 128), bias(max_blocks), output(max_cells, -1234.0f);
    for (int row = 0; row < max_blocks; ++row) {
        bias[row] = (row - 2) * 0.125f;
        for (int k = 0; k < 128; ++k)
            pooled[(size_t) row * 128 + k] = std::sin(0.013f * (row + 1) * (k + 3));
    }
    for (int h = 0; h < 4; ++h)
        for (int k = 0; k < 128; ++k) query[h * 128 + k] = std::cos(0.017f * (h + 2) * (k + 1));

    cudaStream_t stream{};
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    float* d_pooled = device_copy(pooled, stream);
    float* d_query = device_copy(query, stream);
    float* d_bias = device_copy(bias, stream);
    float* d_output = nullptr;
    int32_t* d_step = nullptr;
    ck(cudaMalloc(&d_output, output.size() * sizeof(float)), "output allocation");
    ck(cudaMalloc(&d_step, strata::kernels::kStepCount * sizeof(int32_t)), "step allocation");

    double worst = 0.0;
    for (const int n : {1, 4, 16, 17, 127, 128, 129, 2050, 2051, 2055}) {
        int32_t step[strata::kernels::kStepCount]{};
        strata::kernels::qsa_step_fill(step, n - 1, shapes);
        ck(cudaMemcpyAsync(d_step, step, sizeof(step), cudaMemcpyHostToDevice, stream), "step upload");
        ck(cudaMemcpyAsync(d_output, output.data(), output.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
           "output init");
        strata::kernels::native_qsa_score(d_pooled, d_query, d_bias, shapes, d_step, max_blocks, max_cells,
                                          d_output, stream);
        ck(cudaMemcpyAsync(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
           "output readback");
        ck(cudaStreamSynchronize(stream), "stream sync");
        for (int cell = 0; cell < n; ++cell) {
            const int row = cell / 4;
            const float expected = reference_score(pooled.data(), query.data(), bias.data(), row, n);
            const double error = std::abs((double) output[cell] - expected);
            worst = std::max(worst, error / std::max(1.0, std::abs((double) expected)));
            if (!std::isfinite(output[cell]) || error > 2e-3 * std::max(1.0, std::abs((double) expected))) {
                std::fprintf(stderr, "native_qsa_score_parity: n=%d cell=%d got %.9g expected %.9g\n",
                             n, cell, output[cell], expected);
                return 1;
            }
        }
    }
    selection_check(stream);
    cudaFree(d_step); cudaFree(d_output); cudaFree(d_bias); cudaFree(d_query); cudaFree(d_pooled);
    ck(cudaStreamDestroy(stream), "stream destroy");
    cudaDeviceProp prop{};
    int device = 0;
    ck(cudaGetDevice(&device), "device query");
    ck(cudaGetDeviceProperties(&prop, device), "device properties");
    std::printf("native_qsa_score_parity: OK on %s (sm_%d%d), max relative error %.3g\n",
                prop.name, prop.major, prop.minor, worst);
    return 0;
}
