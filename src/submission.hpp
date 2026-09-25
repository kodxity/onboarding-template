#pragma once

#include <cstddef>
#include <vector>

// Starter Grid for the 2D heat-diffusion problem.
//
// The evaluation harness uses operator() to set initial conditions and to read
// results; it never touches your internal storage. Keep this interface,
// everything else is yours.
class Grid {
   private:
    std::size_t rows_;
    std::size_t cols_;
    std::size_t stride_;
    std::vector<double> data_;

   public:
    Grid(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), stride_(cols), data_(rows * stride_) {}

    double& operator()(std::size_t i, std::size_t j) {
        return data_[i * stride_ + j];
    }
    double operator()(std::size_t i, std::size_t j) const {
        return data_[i * stride_ + j];
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t stride() const { return stride_; }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }
};

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
    const std::size_t rows = old_grid.rows();
    const std::size_t cols = old_grid.cols();

    const double* old_data = old_grid.data();
    double* new_data = new_grid.data();
    const std::size_t old_stride = old_grid.stride();
    const std::size_t new_stride = new_grid.stride();

    for (std::size_t i = 1; i < rows - 1; i++) {
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
        new_grid(0, j) = old_grid(0, j);
        new_grid(rows - 1, j) = old_grid(rows - 1, j);
    }

    for (std::size_t i = 0; i < rows; i++) {
        new_grid(i, 0) = old_grid(i, 0);
        new_grid(i, cols - 1) = old_grid(i, cols - 1);
    }
}