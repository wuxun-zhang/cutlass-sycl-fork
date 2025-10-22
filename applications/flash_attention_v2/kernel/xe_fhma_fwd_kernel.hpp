/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp"
#include "flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp"
#include "flash_attention_v2/kernel/xe_tile_scheduler.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////

struct FMHAProblemShape {
  int batch;
  int num_heads_q, num_heads_kv;
  int seq_len_qo, seq_len_kv;       // -> VariableLen to support variable-length-per-batch cases
  int head_size_qk, head_size_vo;
};

///////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeFMHAFwdKernel {

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;

  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ *Q;
    StrideQ dQ;
    const ElementK *K;
    StrideK dK;
    const ElementV *V;
    StrideV dV;
    ElementO *O;
    StrideO dO;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    return {args.kernel,
            CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{})};
  }

  static bool can_implement(Arguments const &args) {
    return CollectiveMainloop::can_implement(args.mainloop)
        && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const &args) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }


  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf)
  {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    auto &p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b] = tile_scheduler.get_block_coord(); // (Q,V,h,b)
      auto blk_qv = make_coord(blk_q, blk_v);
      int head = head_q / head_group_q;

      const int k_blocks = cute::ceil_div(s.seq_len_kv, get<1>(TileShapeQK{}));

      auto shape_Q = make_shape(s.seq_len_qo, s.head_size_qk, s.num_heads_q,  s.batch);
      auto shape_K = make_shape(s.seq_len_kv, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V = make_shape(s.head_size_vo, s.seq_len_kv, s.num_heads_kv, s.batch);
      auto shape_O = make_shape(s.seq_len_qo, s.head_size_vo, s.num_heads_kv, s.batch);

      auto dcQ = const_cast<ElementQ*>(p.Q);  // de-const these for uniformity
      auto dcK = const_cast<ElementK*>(p.K);
      auto dcV = const_cast<ElementV*>(p.V);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, p.dQ));    // (q,d,h,b)
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, p.dK));    // (k,d,h,b)
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, p.dV));    // (v,k,h,b)
      Tensor O = make_tensor(make_gmem_ptr(p.O), make_layout(shape_O, p.dO));    // (q,v,h,b)

      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;

      // Main loop
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      mainloop(Q(_,_,head_q,idx_b),
               K(_,_,head,idx_b),
               V(_,_,head,idx_b),
               tArA, tA_max, tA_sum,
               blk_qv, 0, k_blocks, k_blocks,
               thr_id);

      if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      epilogue(O(_,_,head_q,idx_b),
               tArA, tA_max, tA_sum,
               blk_qv, thr_id);
    }
  }
};

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class XeFMHAFwdGqaKernel {

public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;

  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  // (1,1,2,4)
  using FragA = typename CollectiveMainloop::FragA;
  using SingleFragA = typename CollectiveMainloop::SingleFragA;
  // (1)
  using FragARow = typename CollectiveMainloop::FragARow;
  // static_assert(is_same_v<decltype(FragA{}.shape()), float>, "dtype mismatch");
  // static_assert(is_same_v<decltype(FragARow{}.shape()), float>, "dtype mismatch");
  // element dtype for MmaPV results
  using ElementA = typename CollectiveMainloop::ElementA;

  // Tile scheduler derived types
  static_assert(is_same_v<TileScheduler_, XeFHMAIndividualTileSchedulerGQA>);
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize = is_empty_v<SharedStorage> ? size_t(0)
                                                                     : sizeof(SharedStorage);
  
  // Important: make sure multiple of 16 element for each copy
  // this is for storing partial results from different KV partitions
  static constexpr int num_elem_per_thead = (size(FragA{}.shape()) + 2 * size(FragARow{}.shape()) + 15) / 16 * 16;

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ *Q;
    StrideQ dQ;
    const ElementK *K;
    StrideK dK;
    const ElementV *V;
    StrideV dV;
    ElementO *O;
    StrideO dO;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
    // workspace for storing partial results of different KV partitions
    ElementA *partial_results_ptr = nullptr;
    // for atomic add
    int32_t *atomic_reduce_cnt_ptr = nullptr;
  };

  //
  // Methods
  //

  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_q;
    int32_t *atomic_reduce_cnt_ptr = reinterpret_cast<int32_t *>(workspace);
    ElementA *partial_results_ptr = reinterpret_cast<ElementA *>(atomic_reduce_cnt_ptr + num_batch_heads);
    return {args.kernel,
            CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.kernel.shape, args.hw_info, TileShapeO{}),
            partial_results_ptr, atomic_reduce_cnt_ptr
          };
  }

  static bool can_implement(Arguments const &args) {
    return CollectiveMainloop::can_implement(args.mainloop)
        && CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const &args) {
    int ws_size = 0;
    // TODO: support more values in future
    int num_partitions = 4; // for 5/1
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_q;
    const int wg_size = SGPerWG::value * intel::sg_size;

    // partial attn outputs, exp sum and max logits
    ws_size += num_partitions * num_batch_heads * wg_size * num_elem_per_thead * sizeof(ElementA);
    // atomic counter
    ws_size += num_batch_heads * sizeof(int32_t);
    return ws_size;
  }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    int num_batch_heads = args.kernel.shape.batch * args.kernel.shape.num_heads_q;
    compat::fill(reinterpret_cast<int32_t*>(workspace), (int32_t)0, num_batch_heads);
    auto partial_ws_count = (get_workspace_size(args) - num_batch_heads * sizeof(int32_t)) / sizeof(ElementA);
    auto* partial_results_ptr = reinterpret_cast<ElementA*>(reinterpret_cast<int32_t*>(workspace) + num_batch_heads);
    compat::fill(partial_results_ptr, (ElementA)0, partial_ws_count);
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(SGPerWG::value * intel::sg_size, 1, 1); }


  template <class Params, class FragA, class FragARow>
  CUTLASS_DEVICE
  void reduce_split2(const Params &params, FragA &out1, FragARow& max_val1, FragARow& exp_sum_val1, FragA &out2, FragARow& max_val2, FragARow& exp_sum_val2) {
    // global max value
    FragARow max_prev1 = max_val1;
    FragARow max_prev2 = max_val2;

    auto scale = params.mainloop.scale;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < max_val1.size(); i++) {
      max_val1(i) = sycl::max(max_val1(i), max_val2(i));
    }

    FragARow rescale1, rescale2;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < max_val1.size(); i++) {
      rescale1(i) = sycl::native::exp2(max_prev1(i) - max_val1(i));
      rescale2(i) = sycl::native::exp2(max_prev2(i) - max_val1(i));
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < exp_sum_val1.size(); i++) {
      exp_sum_val1(i) = exp_sum_val1(i) * rescale1(i) + exp_sum_val2(i) * rescale2(i);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < out1.size(); i++)
      out1(i) = out1(i) * broadcast<0>(rescale1, out1, i) + out2(i) * broadcast<0>(rescale2, out2, i);
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf)
  {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);

    auto &p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int ori_batch_head_wg_id = int(BlockIdxZ());

    int sg_id = thr_id / intel::sg_size;
    int tid_in_sg = thr_id % intel::sg_size;
    int num_batch_heads = s.batch * s.num_heads_q;

    TileScheduler tile_scheduler{params.scheduler};
    const int num_partitions = tile_scheduler.params.num_partitions;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_q, idx_b] = tile_scheduler.get_block_coord(); // (Q,V,h,b)
      auto blk_qv = make_coord(blk_q, blk_v);

      // first group of wg to process first partition of KV seq
      // const bool first_group_wg = ori_batch_head_wg_id < num_batch_heads;
      // id of partition that current wg is processing
      const int partition_id = ori_batch_head_wg_id / num_batch_heads;
      const bool is_leader_wg = partition_id == 0;
      // const int batch_head_wg_id = first_group_wg ? ori_batch_head_wg_id : (ori_batch_head_wg_id - num_batch_heads);
      const int batch_head_wg_id = ori_batch_head_wg_id % num_batch_heads;

      // important: correct q head index
      // head_q = first_group_wg ? head_q : (head_q - s.num_heads_q);
      head_q = head_q - partition_id * s.num_heads_q;
      int head_kv = head_q / head_group_q;

      const int k_blocks = cute::ceil_div(s.seq_len_kv, get<1>(TileShapeQK{}));
      const int num_blocks_per_group = k_blocks / num_partitions;

      auto shape_Q = make_shape(s.seq_len_qo, s.head_size_qk, s.num_heads_q,  s.batch);
      auto shape_K = make_shape(s.seq_len_kv, s.head_size_qk, s.num_heads_kv, s.batch);
      auto shape_V = make_shape(s.head_size_vo, s.seq_len_kv, s.num_heads_kv, s.batch);
      auto shape_O = make_shape(s.seq_len_qo, s.head_size_vo, s.num_heads_kv, s.batch);

      auto dcQ = const_cast<ElementQ*>(p.Q);  // de-const these for uniformity
      auto dcK = const_cast<ElementK*>(p.K);
      auto dcV = const_cast<ElementV*>(p.V);

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), make_layout(shape_Q, p.dQ));    // (q,d,h,b)
      Tensor K = make_tensor(make_gmem_ptr(dcK), make_layout(shape_K, p.dK));    // (k,d,h,b)
      Tensor V = make_tensor(make_gmem_ptr(dcV), make_layout(shape_V, p.dV));    // (v,k,h,b)
      Tensor O = make_tensor(make_gmem_ptr(p.O), make_layout(shape_O, p.dO));    // (q,v,h,b)

      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;

      if (thr_id == 0) {
        // reset atomic counter before computation
        *(params.atomic_reduce_cnt_ptr + batch_head_wg_id) = 0;
      }

      // Main loop
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

      int start_blk = partition_id * num_blocks_per_group;
      int end_blk = (partition_id == (num_partitions -1)) ? k_blocks : (start_blk + num_blocks_per_group);

      mainloop(Q(_,_,head_q,idx_b),
               K(_,_,head_kv,idx_b),
               V(_,_,head_kv,idx_b),
               tArA, tA_max, tA_sum,
               blk_qv, start_blk, end_blk, k_blocks,
               thr_id);

      if (!is_leader_wg) {
        // store partial result: tArA, tA_max and tA_sum
        int offset = partition_id * num_batch_heads * num_elem_per_thead * SGPerWG::value * intel::sg_size
                     + batch_head_wg_id * num_elem_per_thead * SGPerWG::value * intel::sg_size
                     + sg_id * intel::sg_size * num_elem_per_thead
                     + tid_in_sg * num_elem_per_thead;
        Tensor tPartial = make_tensor(params.partial_results_ptr + offset, make_shape(Int<num_elem_per_thead>{}));
        Tensor merged_res = make_tensor<ElementA>(Int<num_elem_per_thead>{});

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < size(FragA{}.shape()); ++i) {
          merged_res(i) = tArA(i);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(FragARow{}.shape()); ++i) {
          merged_res(i + size(FragA{}.shape())) = tA_max(i);
          merged_res(i + 1 + size(FragA{}.shape())) = tA_sum(i);
        }
        copy(merged_res, tPartial);
      }

      // after store, set atomic cnt
      if (thr_id == 0) {
        atomicAdd(params.atomic_reduce_cnt_ptr + batch_head_wg_id, 1);
      }

      // if (first_group_wg) {
      //   // [start_blk, end_blk)
      //   int start_blk = 0, end_blk = num_blocks_per_group;
      //   mainloop(Q(_,_,head_q,idx_b),
      //          K(_,_,head_kv,idx_b),
      //          V(_,_,head_kv,idx_b),
      //          tArA, tA_max, tA_sum,
      //          blk_qv, start_blk, end_blk, k_blocks,
      //          thr_id);

      //   // Note: for first group wg, no need to store partial results into
      //   // workspace since we use such wg to reduce partial results

      //   // after store, set atomic cnt
      //   if (thr_id == 0) {
      //     atomicAdd(params.atomic_reduce_cnt_ptr + batch_head_wg_id, 1);
      //   }
      // } else {
      //   // [start_blk, end_blk)
      //   int start_blk = num_blocks_per_group, end_blk = k_blocks;
      //   mainloop(Q(_,_,head_q,idx_b),
      //          K(_,_,head_kv,idx_b),
      //          V(_,_,head_kv,idx_b),
      //          tArA, tA_max, tA_sum,
      //          blk_qv, start_blk, end_blk, k_blocks,
      //          thr_id);
        
      //   // store partial result: tArA, tA_max and tA_sum
      //   int offset = batch_head_wg_id * num_elem_per_thead * SGPerWG::value * intel::sg_size
      //                + sg_id * intel::sg_size * num_elem_per_thead
      //                + tid_in_sg * num_elem_per_thead;
      //   Tensor tPartial = make_tensor(params.partial_results_ptr + offset, make_shape(Int<num_elem_per_thead>{}));
      //   Tensor merged_res = make_tensor<ElementA>(Int<num_elem_per_thead>{});

      //   CUTLASS_PRAGMA_UNROLL
      //   for(int i = 0; i < size(FragA{}.shape()); ++i) {
      //     merged_res(i) = tArA(i);
      //   }
      //   CUTLASS_PRAGMA_UNROLL
      //   for (int i = 0; i < size(FragARow{}.shape()); ++i) {
      //     merged_res(i + size(FragA{}.shape())) = tA_max(i);
      //     merged_res(i + 1 + size(FragA{}.shape())) = tA_sum(i);
      //   }
      //   copy(merged_res, tPartial);

      //   // after store, set atomic cnt
      //   if (thr_id == 0) {
      //     atomicAdd(params.atomic_reduce_cnt_ptr + batch_head_wg_id, 1);
      //   }
      // }

      // first group wg is responsible for reducing partial results
      if (is_leader_wg) {
        // check atomic to wait for partial results ready
        while(atomicLoad(params.atomic_reduce_cnt_ptr + batch_head_wg_id) != num_partitions) {}

        for (int i = 1; i < num_partitions; ++i) {
          int offset = i * num_batch_heads * SGPerWG::value * intel::sg_size * num_elem_per_thead
                     + batch_head_wg_id * SGPerWG::value * intel::sg_size * num_elem_per_thead
                     + sg_id * intel::sg_size * num_elem_per_thead
                     + tid_in_sg * num_elem_per_thead;
          Tensor tPartial = make_tensor(params.partial_results_ptr + offset, make_shape(Int<num_elem_per_thead>{}));
          Tensor merged_res = make_tensor<ElementA>(Int<num_elem_per_thead>{});
          copy(tPartial, merged_res);

          FragA tArA_2;
          FragARow tA_max_2, tA_sum_2;
          CUTLASS_PRAGMA_UNROLL
          for(int i = 0; i < size(FragA{}.shape()); ++i) {
            tArA_2(i) = merged_res(i);
          }

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(FragARow{}.shape()); ++i) {
            tA_max_2(i) = merged_res(i + size(FragA{}.shape()));
            tA_sum_2(i) = merged_res(i + 1 + size(FragA{}.shape()));
          }

          reduce_split2(params, tArA, tA_max, tA_sum, tArA_2, tA_max_2, tA_sum_2);
        }

        // int offset = batch_head_wg_id * SGPerWG::value * intel::sg_size * num_elem_per_thead
        //              + sg_id * intel::sg_size * num_elem_per_thead
        //              + tid_in_sg * num_elem_per_thead;
        // Tensor tPartial = make_tensor(params.partial_results_ptr + offset, make_shape(Int<num_elem_per_thead>{}));
        // Tensor merged_res = make_tensor<ElementA>(Int<num_elem_per_thead>{});
        // copy(tPartial, merged_res);

        // FragA tArA_2;
        // FragARow tA_max_2, tA_sum_2;
        // CUTLASS_PRAGMA_UNROLL
        // for(int i = 0; i < size(FragA{}.shape()); ++i) {
        //   tArA_2(i) = merged_res(i);
        // }

        // CUTLASS_PRAGMA_UNROLL
        // for (int i = 0; i < size(FragARow{}.shape()); ++i) {
        //   tA_max_2(i) = merged_res(i + size(FragA{}.shape()));
        //   tA_sum_2(i) = merged_res(i + 1 + size(FragA{}.shape()));
        // }

        // reduce_split2(params, tArA, tA_max, tA_sum, tArA_2, tA_max_2, tA_sum_2);

        // require group barrier if using SLM
        if constexpr (!is_empty_v<MainloopSharedStorage> && !is_empty_v<EpilogueSharedStorage>) {
          sycl::group_barrier(get_work_group<3>());
        }

        // Epilogue
        CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
        epilogue(O(_,_,head_q,idx_b),
                tArA, tA_max, tA_sum,
                blk_qv, thr_id);
      }
    }
  }
};

} // namespace cutlass::fmha::kernel
