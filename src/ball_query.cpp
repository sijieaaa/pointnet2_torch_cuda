// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "ball_query.h"
#include "utils.h"

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int num_neighbors, const float *query_xyz,
                                     const float *db_xyz, int *idx);

at::Tensor ball_query(at::Tensor query_xyz, 
                      at::Tensor db_xyz, 
                      const float radius,
                      const int num_neighbors) {
  CHECK_CONTIGUOUS(query_xyz);
  CHECK_CONTIGUOUS(db_xyz);
  CHECK_IS_FLOAT(query_xyz);
  CHECK_IS_FLOAT(db_xyz);

  if (query_xyz.type().is_cuda()) {
    CHECK_CUDA(db_xyz);
  }

  at::Tensor idx =
      torch::zeros({query_xyz.size(0), query_xyz.size(1), num_neighbors},
                   at::device(query_xyz.device()).dtype(at::ScalarType::Int));

  if (query_xyz.type().is_cuda()) {
    query_ball_point_kernel_wrapper(db_xyz.size(0), db_xyz.size(1), query_xyz.size(1),
                                    radius, num_neighbors, query_xyz.data<float>(),
                                    db_xyz.data<float>(), idx.data<int>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return idx;
}
