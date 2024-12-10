// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

// input: query_xyz(b, m, 3) db_xyz(b, n, 3)
// output: idx(b, m, num_neighbors)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int num_neighbors,
                                        const float *__restrict__ query_xyz,
                                        const float *__restrict__ db_xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  db_xyz += batch_index * n * 3;
  query_xyz += batch_index * m * 3;
  idx += m * num_neighbors * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = query_xyz[j * 3 + 0];
    float new_y = query_xyz[j * 3 + 1];
    float new_z = query_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < num_neighbors; ++k) {
      float x = db_xyz[k * 3 + 0];
      float y = db_xyz[k * 3 + 1];
      float z = db_xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < num_neighbors; ++l) {
            idx[j * num_neighbors + l] = k;
          }
        }
        idx[j * num_neighbors + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_wrapper(int b, int n, int m, float radius,
                                     int num_neighbors, const float *query_xyz,
                                     const float *db_xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, num_neighbors, query_xyz, db_xyz, idx);

  CUDA_CHECK_ERRORS();
}
