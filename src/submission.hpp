#pragma once

#include <cassert>
#include <cstddef>
#include <vector>

struct Dim {
    std::size_t rows;
    std::size_t cols;

    constexpr bool operator==(const Dim& other) const noexcept {
        return rows == other.rows && cols == other.cols;
    }
};

template <typename T>
struct View {
    T* cells;
    Dim dim;
    std::size_t stride;

    T& operator()(std::size_t row, std::size_t col) const {
        return cells[row * stride + col];
    }
};

using GridView = View<double>;
using ConstGridView = View<const double>;

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
   private:
    Dim dim_;
    std::size_t stride_;
    std::vector<double> data_;

   public:
    Grid(std::size_t rows, std::size_t cols)
        : dim_{rows, cols}, stride_(cols), data_(rows * stride_) {}

    double& operator()(std::size_t i, std::size_t j) {
        return data_[i * stride_ + j];
    }
    double operator()(std::size_t i, std::size_t j) const {
        return data_[i * stride_ + j];
    }

    Dim dim() const { return dim_; }
    std::size_t rows() const { return dim_.rows; }
    std::size_t cols() const { return dim_.cols; }
    std::size_t stride() const { return stride_; }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    GridView view() { return {data_.data(), dim_, stride_}; }
    ConstGridView view() const { return {data_.data(), dim_, stride_}; }
};

inline void solve(ConstGridView old_view, GridView new_view) {
    const std::size_t rows = old_view.dim.rows;
    const std::size_t cols = old_view.dim.cols;

    if (rows == 0 || cols == 0) return;

    const double* __restrict__ old_data = old_view.cells;
    double* __restrict__ new_data = new_view.cells;
    const std::size_t old_stride = old_view.stride;
    const std::size_t new_stride = new_view.stride;

#pragma omp parallel for schedule(static)
    for (std::size_t i = 1; i < rows - 1; i++) {
#pragma omp simd
        for (std::size_t j = 1; j < cols - 1; j++) {
            const std::size_t old_index = i * old_stride + j;
            const std::size_t new_index = i * new_stride + j;
            new_data[new_index] =
                0.5 * old_data[old_index] +
                0.125 * (old_data[old_index - old_stride] +
                         old_data[old_index + old_stride] +
                         old_data[old_index - 1] + old_data[old_index + 1]);
        }
    }

    for (std::size_t j = 0; j < cols; j++) {
        new_data[j] = old_data[j];
        new_data[(rows - 1) * new_stride + j] =
            old_data[(rows - 1) * old_stride + j];
    }

    for (std::size_t i = 0; i < rows; i++) {
        new_data[i * new_stride] = old_data[i * old_stride];
        new_data[i * new_stride + cols - 1] =
            old_data[i * old_stride + cols - 1];
    }
}

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
inline void apply_stencil(const Grid& old_grid, Grid& new_grid) {
    assert(&old_grid != &new_grid &&
           "old_grid and new_grid must not share the same memory");
    assert(old_grid.dim() == new_grid.dim() &&
           "old_grid and new_grid must have the same dimensions");

    solve(old_grid.view(), new_grid.view());
}
