#pragma once

#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <new>
#include <type_traits>
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
    AlignedAllocator(const AlignedAllocator<U, Align> &) noexcept {}

    [[nodiscard]] T *allocate(std::size_t count) {
        if (count == 0)
            return nullptr;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        return static_cast<T *>(
            ::operator new(count * sizeof(T), std::align_val_t(Align)));
    }

    void deallocate(T *ptr, std::size_t) noexcept {
        ::operator delete(ptr, std::align_val_t(Align));
    }
};

template <class T, class U, std::size_t Align>
bool operator==(const AlignedAllocator<T, Align> &,
                const AlignedAllocator<U, Align> &) {
    return true;
}

template <class T, class U, std::size_t Align>
bool operator!=(const AlignedAllocator<T, Align> &,
                const AlignedAllocator<U, Align> &) {
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
        : rows_(rows), cols_(cols), stride_((cols + 3) & ~std::size_t(3)),
          data_(rows * stride_) {}

    double &operator()(std::size_t i, std::size_t j) {
        return data_[i * stride_ + j];
    }

    double operator()(std::size_t i, std::size_t j) const {
        return data_[i * stride_ + j];
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    double *data() { return data_.data(); }
    const double *data() const { return data_.data(); }
    std::size_t stride() const { return stride_; }
};

inline void copy_boundaries(const Grid &old_grid, Grid &new_grid) {
    const std::size_t rows = old_grid.rows();
    const std::size_t cols = old_grid.cols();
    if (rows == 0 || cols == 0)
        return;

    const std::size_t stride = old_grid.stride();
    const double *old_data = old_grid.data();
    double *new_data = new_grid.data();

    std::memcpy(new_data, old_data, cols * sizeof(double));
    if (rows > 1) {
        std::memcpy(new_data + (rows - 1) * stride,
                    old_data + (rows - 1) * stride, cols * sizeof(double));
    }

    for (std::size_t i = 1; i + 1 < rows; ++i) {
        new_data[i * stride] = old_data[i * stride];
        new_data[i * stride + cols - 1] =
            old_data[i * stride + cols - 1];
    }
}

void apply_stencil(const Grid &old_grid, Grid &new_grid) {
    const std::size_t rows = old_grid.rows();
    const std::size_t cols = old_grid.cols();
    const std::size_t stride = old_grid.stride();

    const double *__restrict__ prev_grid = old_grid.data();
    double *__restrict__ curr_grid = new_grid.data();

    copy_boundaries(old_grid, new_grid);
    if (rows < 3 || cols < 3)
        return;

    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d eighth = _mm256_set1_pd(0.125);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 1; i < rows - 1; i++) {
        const double *__restrict__ prev_row = prev_grid + (i - 1) * stride;
        const double *__restrict__ curr_row = prev_grid + i * stride;
        const double *__restrict__ next_row = prev_grid + (i + 1) * stride;
        double *__restrict__ res = curr_grid + i * stride;

        std::size_t j = 1;

        // Keep the first three interior cells scalar so j=4 is aligned.
        for (; j < 4 && j < cols - 1; ++j) {
            res[j] = 0.5 * curr_row[j] +
                     0.125 * (prev_row[j] + next_row[j] +
                              curr_row[j - 1] + curr_row[j + 1]);
        }

        if (j + 4 <= cols - 1) {
            __m256d left_window = _mm256_loadu_pd(curr_row + j - 1);

            for (; j + 4 <= cols - 1; j += 4) {
                const __m256d right_window =
                    _mm256_loadu_pd(curr_row + j + 3);

                const __m256d center = _mm256_blend_pd(
                    _mm256_permute4x64_pd(
                        left_window, _MM_SHUFFLE(0, 3, 2, 1)),
                    _mm256_permute4x64_pd(right_window, 0x00), 0x8);
                const __m256d left = left_window;
                const __m256d right = _mm256_blend_pd(
                    _mm256_permute4x64_pd(
                        left_window, _MM_SHUFFLE(3, 2, 3, 2)),
                    _mm256_permute4x64_pd(
                        right_window, _MM_SHUFFLE(1, 0, 1, 0)),
                    0xc);

                const __m256d vertical = _mm256_add_pd(
                    _mm256_loadu_pd(prev_row + j),
                    _mm256_loadu_pd(next_row + j));
                const __m256d horizontal = _mm256_add_pd(left, right);
                const __m256d result = _mm256_fmadd_pd(
                    _mm256_add_pd(vertical, horizontal), eighth,
                    _mm256_mul_pd(center, half));

                _mm256_store_pd(res + j, result);
                left_window = right_window;
            }
        }

        for (; j < cols - 1; ++j) {
            res[j] = 0.5 * curr_row[j] +
                     0.125 * (prev_row[j] + next_row[j] +
                              curr_row[j - 1] + curr_row[j + 1]);
        }
    }
}
