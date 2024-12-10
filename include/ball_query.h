// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <torch/extension.h>

at::Tensor ball_query(at::Tensor query_xyz, 
                        at::Tensor db_xyz, 
                        const float radius,
                      const int num_neighbors);
