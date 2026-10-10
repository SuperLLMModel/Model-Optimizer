# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.distributed as dist
from _test_utils.torch.distributed.utils import spawn_multiprocess_job
from _test_utils.torch.quantization.models import SimpleLinear
from torch.nn.parallel import DistributedDataParallel

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils import unwrap_model
from modelopt.torch.utils.distributed import DistributedProcessGroup


def _test_data_parallel_helper(rank, size):
    model = DistributedDataParallel(SimpleLinear())
    unwrapped_model = unwrap_model(model)
    calib_data = [unwrapped_model.get_input() for _ in range(2)]

    def forward_loop(model):
        for data in calib_data:
            model(data)

    unwrapped_model = mtq.quantize(unwrapped_model, mtq.INT8_DEFAULT_CFG, forward_loop)

    for name, module in unwrapped_model.named_modules():
        if isinstance(module, TensorQuantizer) and "output_quantizer" not in name:
            amax = module.amax.clone()
            dist.all_reduce(amax, op=dist.ReduceOp.MAX)
            assert torch.allclose(amax, module.amax)
    dist.destroy_process_group()


def test_data_parallel(skip_on_windows):
    spawn_multiprocess_job(2, _test_data_parallel_helper, backend="gloo")


def _test_uneven_amax_sync_helper(rank, size):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=8))
    if rank == 0:
        quantizer._set_buffer("_amax", torch.tensor(7.0))
    quantizer.sync_amax_across_distributed_group(DistributedProcessGroup(dist.group.WORLD))
    assert quantizer.amax.item() == 7.0
    dist.destroy_process_group()


def test_amax_sync_with_a_rank_that_never_calibrated(skip_on_windows):
    """The rank without an amax still joins the sync, contributing a zero on the group's device."""
    spawn_multiprocess_job(2, _test_uneven_amax_sync_helper, backend="gloo")
