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
  std::vector<double> data_;


public:
  Grid(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(rows*cols) {

  }

  double& operator()(std::size_t i, std::size_t j) {
    return data_[i*cols_+j];
  }
  double  operator()(std::size_t i, std::size_t j) const {
    return data_[i*cols_+j];
  }

  std::size_t rows() const {
      return rows_;
  }

  std::size_t cols() const {
      return cols_;
  }
  double* data() {
      return data_.data();
  }

  const double* data() const {
      return data_.data();
  }
};  

// Apply the five-point stencil over all interior points, copying the boundary
// values unchanged from old_grid to new_grid. Implement your solution here.
void apply_stencil(const Grid& old_grid, Grid& new_grid) {
  const std::size_t rows = old_grid.rows();
  const std::size_t cols = old_grid.cols();

  const double* __restrict__ prev_grid = old_grid.data();
  double* __restrict__ curr_grid = new_grid.data();

  for (std::size_t i = 1; i < rows-1; i++) {
    const double* __restrict__ prev_row = prev_grid+(i-1)*cols;
    const double* __restrict__ curr_row = prev_grid+i*cols;
    const double* __restrict__ next_row = prev_grid+(i+1)*cols;
    double* __restrict__ res = curr_grid+i*cols;
    for (std::size_t j = 1; j < cols-1; j++) {
      res[j] = 0.5 * curr_row[j] + 0.125 * (prev_row[j] + next_row[j] + curr_row[j-1] + curr_row[j+1]);
    }
  }

  for (std::size_t j = 0; j < cols; j++) {
      curr_grid[j] = prev_grid[j];
      curr_grid[(rows-1)*cols+j] = prev_grid[(rows-1)*cols+j];
  }

  for (std::size_t i = 0; i < rows; i++) {
    curr_grid[i*cols] = prev_grid[i*cols];
    curr_grid[i*cols+cols-1] = prev_grid[i*cols+cols-1];
  }
}
