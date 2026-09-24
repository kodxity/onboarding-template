#pragma once
// Tuned for: g++ -std=gnu++17 -O3 -march=x86-64-v3 -fopenmp
// x86-64-v3 guarantees AVX2+FMA are always present at runtime on this build,
// so there is no need to runtime-check for them or compile them behind a
// target() attribute -- they're just "the normal compiled code". AVX-512F is
// NOT guaranteed by v3, so it stays behind a runtime cpuid check with a
// fallback to the (always-correct) AVX2 path, never to scalar.
#include <immintrin.h>
#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

template <class T, std::size_t Align = 64>
struct AlignedAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

    AlignedAllocator() noexcept = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        return static_cast<T*>(
            ::operator new(n * sizeof(T), std::align_val_t(Align)));
    }
    void deallocate(T* p, std::size_t) noexcept {
        ::operator delete(p, std::align_val_t(Align));
    }
};
template <class T, class U, std::size_t A>
bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) {
    return true;
}
template <class T, class U, std::size_t A>
bool operator!=(const AlignedAllocator<T, A>&, const AlignedAllocator<U, A>&) {
    return false;
}

class Grid {
   private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t stride_;
    std::vector<double, AlignedAllocator<double, 64>> data_;

   public:
    Grid(std::size_t rows, std::size_t cols)
        : rows_(rows),
          cols_(cols),
          stride_((cols + 7) & ~std::size_t(7)),
          data_(rows * stride_) {}

    double& operator()(std::size_t i, std::size_t j) {
        return data_[i * stride_ + j];
    }
    double operator()(std::size_t i, std::size_t j) const {
        return data_[i * stride_ + j];
    }
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t stride() const noexcept { return stride_; }
    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }
};

inline constexpr std::size_t kMinInteriorDim = 3;
inline constexpr std::size_t kParallelRowThreshold = 64;

inline void copy_boundaries(const Grid& old_grid, Grid& new_grid) {
    const std::size_t rows = old_grid.rows();
    const std::size_t cols = old_grid.cols();
    const std::size_t stride = old_grid.stride();
    const double* prev = old_grid.data();
    double* curr = new_grid.data();

    std::memcpy(curr, prev, cols * sizeof(double));
    if (rows > 1)
        std::memcpy(curr + (rows - 1) * stride, prev + (rows - 1) * stride,
                    cols * sizeof(double));
    for (std::size_t i = 1; i < rows - 1; ++i) {
        curr[i * stride] = prev[i * stride];
        curr[i * stride + cols - 1] = prev[i * stride + cols - 1];
    }
}

inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
    const std::size_t rows = old_grid.rows();
    const std::size_t cols = old_grid.cols();
    const std::size_t stride = old_grid.stride();
    const double* __restrict__ prev_grid = old_grid.data();
    double* __restrict__ curr_grid = new_grid.data();

    copy_boundaries(old_grid, new_grid);
    if (rows < kMinInteriorDim || cols < kMinInteriorDim) return;

    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d eighth = _mm256_set1_pd(0.125);

#pragma omp parallel for schedule(static) if (rows > kParallelRowThreshold)
    for (std::size_t i = 1; i < rows - 1; i++) {
        const double* __restrict__ prev_row = prev_grid + (i - 1) * stride;
        const double* __restrict__ center_row = prev_grid + i * stride;
        const double* __restrict__ next_row = prev_grid + (i + 1) * stride;
        double* __restrict__ res = curr_grid + i * stride;

        std::size_t j = 1;

        for (; j + 8 <= cols - 1; j += 8) {
            __m256d p0 = _mm256_loadu_pd(&prev_row[j]);
            __m256d n0 = _mm256_loadu_pd(&next_row[j]);
            __m256d l0 = _mm256_loadu_pd(&center_row[j - 1]);
            __m256d r0 = _mm256_loadu_pd(&center_row[j + 1]);
            __m256d c0 = _mm256_loadu_pd(&center_row[j]);

            __m256d p1 = _mm256_loadu_pd(&prev_row[j + 4]);
            __m256d n1 = _mm256_loadu_pd(&next_row[j + 4]);
            __m256d l1 = _mm256_loadu_pd(&center_row[j + 3]);
            __m256d r1 = _mm256_loadu_pd(&center_row[j + 5]);
            __m256d c1 = _mm256_loadu_pd(&center_row[j + 4]);

            __m256d sum0 =
                _mm256_add_pd(_mm256_add_pd(p0, n0), _mm256_add_pd(l0, r0));
            __m256d res0 =
                _mm256_fmadd_pd(sum0, eighth, _mm256_mul_pd(c0, half));

            __m256d sum1 =
                _mm256_add_pd(_mm256_add_pd(p1, n1), _mm256_add_pd(l1, r1));
            __m256d res1 =
                _mm256_fmadd_pd(sum1, eighth, _mm256_mul_pd(c1, half));

            _mm256_storeu_pd(&res[j], res0);
            _mm256_storeu_pd(&res[j + 4], res1);
        }
        for (; j + 4 <= cols - 1; j += 4) {
            __m256d p = _mm256_loadu_pd(&prev_row[j]);
            __m256d n = _mm256_loadu_pd(&next_row[j]);
            __m256d l = _mm256_loadu_pd(&center_row[j - 1]);
            __m256d r = _mm256_loadu_pd(&center_row[j + 1]);
            __m256d c = _mm256_loadu_pd(&center_row[j]);
            __m256d sum =
                _mm256_add_pd(_mm256_add_pd(p, n), _mm256_add_pd(l, r));
            __m256d result =
                _mm256_fmadd_pd(sum, eighth, _mm256_mul_pd(c, half));
            _mm256_storeu_pd(&res[j], result);
        }
        for (; j < cols - 1; j++) {
            res[j] = 0.5 * center_row[j] +
                     0.125 * (prev_row[j] + next_row[j] + center_row[j - 1] +
                              center_row[j + 1]);
        }
    }
}
