#pragma once

#include <cstddef>
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

void apply_stencil(const Grid &old_grid, Grid &new_grid) {
    const std::size_t rows = old_grid.rows();
    const std::size_t cols = old_grid.cols();
    const std::size_t stride = old_grid.stride();

    const double *__restrict__ prev_grid = old_grid.data();
    double *__restrict__ curr_grid = new_grid.data();

#pragma omp parallel for schedule(static)
    for (std::size_t i = 1; i < rows - 1; i++) {
        const double *__restrict__ prev_row = prev_grid + (i - 1) * stride;
        const double *__restrict__ curr_row = prev_grid + i * stride;
        const double *__restrict__ next_row = prev_grid + (i + 1) * stride;
        double *__restrict__ res = curr_grid + i * stride;

#pragma omp simd
        for (std::size_t j = 1; j < cols - 1; j++) {
            res[j] = 0.5 * curr_row[j] +
                     0.125 * (prev_row[j] + next_row[j] + curr_row[j - 1] +
                              curr_row[j + 1]);
        }
    }

    for (std::size_t j = 0; j < cols; j++) {
        curr_grid[j] = prev_grid[j];
        curr_grid[(rows - 1) * stride + j] =
            prev_grid[(rows - 1) * stride + j];
    }

    for (std::size_t i = 0; i < rows; i++) {
        curr_grid[i * stride] = prev_grid[i * stride];
        curr_grid[i * stride + cols - 1] =
            prev_grid[i * stride + cols - 1];
    }
}
