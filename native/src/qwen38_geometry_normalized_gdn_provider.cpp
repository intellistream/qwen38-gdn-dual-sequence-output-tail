#include "statecentric/qwen38_provider_api_v1.h"

#ifdef STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1
#include "statecentric/completed_buffer_pool.h"
#if defined(STATECENTRIC_DEVICE_ALLOCATION_DIAGNOSTICS)
#error "The prefill tensor candidate must not include allocation observers"
#endif
#endif

#ifdef STATECENTRIC_DEVICE_ALLOCATION_DIAGNOSTICS
#include "statecentric/device_allocation_diagnostics.h"
#ifdef STATECENTRIC_DEVICE_ALLOCATION_CATEGORIES
#include "statecentric/device_allocation_categories.h"
#endif
namespace {
#ifdef STATECENTRIC_DEVICE_ALLOCATION_CATEGORIES
statecentric::DeviceAllocationCategoryLedger event_allocation_ledger;
#else
statecentric::DeviceAllocationLedger event_allocation_ledger;
#endif
}
extern "C" int statecentric_device_allocation_diagnostics_v1(
    std::uint32_t version, std::uint32_t size,
    statecentric::DeviceAllocationDiagnosticsV1* output) {
#ifdef STATECENTRIC_DEVICE_ALLOCATION_CATEGORIES
  return statecentric::ReadCategorizedAllocationTotal(event_allocation_ledger,
      statecentric::DeviceAllocationScope::kGeometryGdnEventOwned, version, size, output);
#else
  return statecentric::ReadDeviceAllocationDiagnostics(event_allocation_ledger,
      statecentric::DeviceAllocationScope::kGeometryGdnEventOwned,
      version, size, output);
#endif
}
#ifdef STATECENTRIC_DEVICE_ALLOCATION_CATEGORIES
extern "C" int statecentric_device_allocation_categories_v1(
    std::uint32_t version, std::uint32_t size,
    statecentric::DeviceAllocationCategoriesV1* output) {
  return statecentric::ReadDeviceAllocationCategories(event_allocation_ledger,
      statecentric::DeviceAllocationScope::kGeometryGdnEventOwned, version, size, output);
}
#endif
#endif

#ifdef STATECENTRIC_DEVICE_ALLOCATION_CATEGORIES
#define GDN_WORKSPACE(event, bytes, category) \
  Allocate(event, bytes, statecentric::DeviceAllocationCategory::category)
#elif defined(STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1)
#define GDN_WORKSPACE(event, bytes, category) AllocatePrivate(event, bytes)
#else
#define GDN_WORKSPACE(event, bytes, category) Allocate(event, bytes)
#endif
#include "statecentric/qwen38_gdn_batched_projection_policy.h"
#include "statecentric/qwen38_gdn_dual_sequence_tail_policy.h"
#include "statecentric/qwen38_gdn_event_pool_policy.h"
#include "statecentric/qwen38_gdn_graph_event_policy.h"
#include "statecentric/qwen38_gdn_geometry_policy.h"
#include "statecentric/qwen38_gdn_sequence_stream_plan.h"
#include "statecentric/qwen38_triton_aot.h"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_batch_matmul.h>
#include <aclnnop/aclnn_causal_conv1d.h>
#include <aclnnop/aclnn_chunk_fwd_o.h>
#include <aclnnop/aclnn_chunk_gated_delta_rule_fwd_h.h>
#include <aclnnop/aclnn_fused_gdn_gating.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_recurrent_gated_delta_rule.h>
#include <aclnnop/aclnn_rms_norm.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_slice.h>
#include <aclnnop/aclnn_split_with_size.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include <runtime/runtime/rt.h>

struct statecentric_qwen38_split_cache_v1 {
  aclTensor* input = nullptr;
  aclTensor* query = nullptr;
  aclTensor* key = nullptr;
  aclTensor* value = nullptr;
  aclIntArray* split_sizes = nullptr;
  aclTensorList* outputs = nullptr;
  aclOpExecutor* executor = nullptr;
  std::uint64_t workspace_bytes = 0;
  bool in_use = false;

  ~statecentric_qwen38_split_cache_v1() {
    if (executor != nullptr) (void)aclDestroyAclOpExecutor(executor);
    if (outputs != nullptr) (void)aclDestroyTensorList(outputs);
    if (split_sizes != nullptr) (void)aclDestroyIntArray(split_sizes);
    if (value != nullptr) (void)aclDestroyTensor(value);
    if (key != nullptr) (void)aclDestroyTensor(key);
    if (query != nullptr) (void)aclDestroyTensor(query);
    if (input != nullptr) (void)aclDestroyTensor(input);
  }
};

constexpr std::size_t kQwen38SynchronizationEventsPerBundle = 7;

struct statecentric_qwen38_synchronization_event_bundle_v1 {
  std::array<aclrtEvent, kQwen38SynchronizationEventsPerBundle> events{};
  bool in_use = false;
};

struct statecentric_qwen38_provider_handle_v1 {
#ifdef STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1
  std::unique_ptr<statecentric::CompletedBufferPool> prefill_tensor_pool;
#endif
  const statecentric_qwen38_host_v1* host = nullptr;
  std::uint32_t rank = 0;
  std::uint32_t max_tokens = 0;
  std::uint32_t qkv = 0;
  std::uint32_t value = 0;
  std::uint32_t gate_heads = 0;
  std::uint32_t query_heads = 0;
  std::uint32_t value_heads = 0;
  std::uint32_t prefill_qkv_split_variant = 0;
  bool recurrent_prefill_always_recurrent = false;
  bool wy_outputs_bhtd = false;
  bool batched_projection_v1 = false;
  bool parallel_batched_projection_v2 = false;
  bool synchronization_event_pool_v3 = false;
  bool prune_parent_projection_waits_v4 = false;
  bool reusable_event_ex_v5 = false;
  bool event_pool_telemetry_v6 = false;
  bool event_pool_runtime_telemetry_v7 = false;
  bool event_pool_atomic_runtime_telemetry_v8 = false;
  std::size_t synchronization_event_pool_active_bundles_v9 =
      statecentric::kQwen38SynchronizationEventBundleCount;
  bool dual_sequence_tail_v10 = false;
  bool projection_only_tail_v11 = false;
  bool packed_decode_recurrent_v12 = false;
  bool lane_local_packed_recurrent_v13 = false;
  bool lane_anchored_packed_recurrent_v14 = false;
  bool graph_captured_lane_events_v15 = false;
  bool graph_single_follower_v16 = false;
  bool eager_single_follower_v17 = false;
  bool decode_flat_output_projection_v18 = false;
  bool parent_local_decode_tail_v19 = false;
  std::atomic<std::uint64_t> packed_decode_submit_calls_v12{0};
  std::atomic<std::uint64_t> packed_decode_admitted_calls_v12{0};
  std::atomic<bool> packed_decode_admission_emitted_v12{false};
  std::atomic<std::uint64_t> lane_anchored_submit_calls_v14{0};
  std::atomic<std::uint64_t> lane_anchored_admitted_calls_v14{0};
  std::atomic<bool> lane_anchored_admission_emitted_v14{false};
  std::atomic<std::uint64_t> graph_captured_lane_admissions_v15{0};
  std::atomic<bool> graph_captured_lane_admission_emitted_v15{false};
  std::atomic<std::uint64_t> graph_single_follower_admissions_v16{0};
  std::atomic<bool> graph_single_follower_admission_emitted_v16{false};
  std::atomic<std::uint64_t> eager_single_follower_admissions_v17{0};
  std::atomic<bool> eager_single_follower_admission_emitted_v17{false};
  std::atomic<std::uint64_t> decode_flat_output_admissions_v18{0};
  std::atomic<bool> decode_flat_output_admission_emitted_v18{false};
  std::atomic<std::uint64_t> parent_local_decode_tail_admissions_v19{0};
  std::atomic<bool> parent_local_decode_tail_admission_emitted_v19{false};
  void* l2_gamma = nullptr;
  std::array<void*, 128> actual_lengths{};
  std::array<void*, 8> state_indices{};
  void* packed_decode_actual_lengths = nullptr;
  void* packed_decode_state_indices = nullptr;
  void* chunk_indices = nullptr;
  void* ffts = nullptr;
  std::uint32_t ai_cores = 0;
  std::array<aclrtStream, 4> sequence_streams{};
  std::mutex sequence_stream_mutex;
  std::mutex synchronization_event_pool_mutex;
  std::array<statecentric_qwen38_synchronization_event_bundle_v1,
             statecentric::kQwen38SynchronizationEventBundleCount>
      synchronization_event_bundles{};
  std::size_t synchronization_event_pool_cursor = 0;
  std::uint64_t synchronization_event_pool_acquire_attempts = 0;
  std::uint64_t synchronization_event_pool_acquisitions = 0;
  std::uint64_t synchronization_event_pool_fallbacks = 0;
  std::uint64_t synchronization_event_pool_releases = 0;
  std::size_t synchronization_event_pool_peak_in_use = 0;
  std::array<std::uint64_t,
             statecentric::kQwen38SynchronizationEventBundleCount>
      synchronization_event_pool_bundle_leases{};
  std::mutex split_cache_mutex;
  std::array<std::vector<std::unique_ptr<statecentric_qwen38_split_cache_v1>>,
             129>
      split_caches;
  std::unique_ptr<statecentric::Qwen38TritonAotKernel> cumsum_kernel;
  std::unique_ptr<statecentric::Qwen38TritonAotKernel> kkt_kernel;
  std::unique_ptr<statecentric::Qwen38TritonAotKernel> solve_kernel;
  std::unique_ptr<statecentric::Qwen38TritonAotKernel> merge_kernel;
  std::unique_ptr<statecentric::Qwen38TritonAotKernel> wy_kernel;
  std::unique_ptr<statecentric::Qwen38TritonAotKernel> post_conv_pack_kernel;
};

struct statecentric_qwen38_event_v1 {
#ifdef STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1
  statecentric::CompletedBufferPool* tensor_pool = nullptr;
  std::vector<statecentric::CompletedBufferPool::Lease> tensor_leases;
#endif
  aclrtStream stream = nullptr;
  std::uint64_t submitted_ns = 0;
  bool complete = false;
  std::vector<void*> allocations;
  std::vector<statecentric_qwen38_split_cache_v1*> split_caches;
  std::vector<statecentric_qwen38_event_v1*> sequence_events;
  std::vector<aclrtEvent> synchronization_events;
  bool model_ri_cross_stream_capture = false;
  statecentric_qwen38_synchronization_event_bundle_v1*
      synchronization_event_bundle = nullptr;
};

struct statecentric_qwen38_precomputed_projection_v1 {
  std::uint64_t magic = 0;
  std::uint32_t tokens = 0;
  std::uint32_t reserved = 0;
  void* qkv = nullptr;
  void* z = nullptr;
  void* recurrent_output = nullptr;
  void* gated_output = nullptr;
  void* query_norm = nullptr;
  void* key_norm = nullptr;
  void* value = nullptr;
  void* beta = nullptr;
  void* gate = nullptr;
  std::uint32_t defer_output_tail = 0;
  std::uint32_t defer_output_projection = 0;
  std::uint32_t use_precomputed_projection = 0;
  std::uint32_t defer_recurrent = 0;
};

namespace {

constexpr std::int64_t kHidden = 5120;
constexpr std::int64_t kHeadDim = 128;
constexpr std::int64_t kConvHistory = 3;
constexpr std::uint32_t kRecurrentKernelTokens = 8;
constexpr std::uint32_t kChunkKernelMinimumTokens = 9;
constexpr std::uint32_t kMaximumSubmissionTokens = 128;
constexpr std::size_t kMaximumConcurrentSplitCachesPerGeometry = 4;
constexpr std::int64_t kChunkSize = 64;
constexpr std::uint64_t kPrecomputedProjectionMagic = 0x51474E4250524F4AULL;
constexpr char kTp4CumsumSha256[] =
    "4fad94f4b0ce14a0b8a090a7d2ddfbcd335629f2e68cedf147d562a3ee4b070e";
constexpr char kTp4KktSha256[] =
    "20ebd60421e49878f7fc1e0babf107ee405dc80a9d691c76622f1584bb9d8ac6";
constexpr char kTp4SolveSha256[] =
    "dd4e3c066cf68a329effcf3bbfb1b4704f3e8bc7337cdc2a5def7470cba2d01d";
constexpr char kTp4MergeSha256[] =
    "1ccfe0c5f00e1ca3f95f1a7a038788816559f72ca699dbe1ee2cc59107823e5c";
constexpr char kTp4WySha256[] =
    "e884188b30ab07d36cee7f6fa19eb9b6a0821b11128d176f09ef8eecb5912361";
constexpr char kTp2CumsumSha256[] =
    "a929fb83c26017f6630a19d60d6a988493e429be78db11347c2bead092d13954";
constexpr char kTp2KktSha256[] =
    "fcd22e98e78bb4b53acd626f2ecb5ff15e6b45b4097e0bc71d307c57ad2fdba0";
constexpr char kTp2SolveSha256[] =
    "dd4e3c066cf68a329effcf3bbfb1b4704f3e8bc7337cdc2a5def7470cba2d01d";
constexpr char kTp2MergeSha256[] =
    "1ccfe0c5f00e1ca3f95f1a7a038788816559f72ca699dbe1ee2cc59107823e5c";
constexpr char kTp2WySha256[] =
    "e884188b30ab07d36cee7f6fa19eb9b6a0821b11128d176f09ef8eecb5912361";
constexpr char kTp2WyBhtdSha256[] =
    "8c3d2ccd3114ee2421b40300459e3d4b8f41778d7a64f6b1f20ed56d41e38ac0";
constexpr char kTp2PostConvPackSha256[] =
    "209c28e5af7bcc58170d182e4c7c0579130c9010f891b76ce18b0289850a7e22";

struct __attribute__((packed)) CumsumArguments {
  void* ffts __attribute__((aligned(8)));
  void* sync __attribute__((aligned(8)));
  void* workspace __attribute__((aligned(8)));
  void* input __attribute__((aligned(8)));
  void* output __attribute__((aligned(8)));
  void* cumulative __attribute__((aligned(8)));
  void* blocks __attribute__((aligned(8)));
  std::int32_t tokens __attribute__((aligned(4)));
  std::int32_t grid_x __attribute__((aligned(4)));
  std::int32_t grid_y __attribute__((aligned(4)));
  std::int32_t grid_z __attribute__((aligned(4)));
};
static_assert(sizeof(CumsumArguments) == 72);

struct __attribute__((packed)) PostConvPackArguments {
  void* ffts __attribute__((aligned(8)));
  void* sync __attribute__((aligned(8)));
  void* workspace __attribute__((aligned(8)));
  void* mixed_qkv __attribute__((aligned(8)));
  void* query __attribute__((aligned(8)));
  void* key __attribute__((aligned(8)));
  void* value __attribute__((aligned(8)));
  std::int32_t mixed_qkv_token_stride __attribute__((aligned(4)));
  std::int32_t query_token_stride __attribute__((aligned(4)));
  std::int32_t key_token_stride __attribute__((aligned(4)));
  std::int32_t value_token_stride __attribute__((aligned(4)));
  std::int32_t tokens __attribute__((aligned(4)));
  std::int32_t grid_x __attribute__((aligned(4)));
  std::int32_t grid_y __attribute__((aligned(4)));
  std::int32_t grid_z __attribute__((aligned(4)));
};
static_assert(sizeof(PostConvPackArguments) == 88);

struct __attribute__((packed)) KktArguments {
  void* ffts __attribute__((aligned(8)));
  void* sync __attribute__((aligned(8)));
  void* workspace __attribute__((aligned(8)));
  void* key __attribute__((aligned(8)));
  void* beta __attribute__((aligned(8)));
  void* gate __attribute__((aligned(8)));
  void* output __attribute__((aligned(8)));
  void* cumulative __attribute__((aligned(8)));
  void* chunks __attribute__((aligned(8)));
  std::int32_t tokens __attribute__((aligned(4)));
  std::int32_t batch __attribute__((aligned(4)));
  std::int32_t grid_x __attribute__((aligned(4)));
  std::int32_t grid_y __attribute__((aligned(4)));
  std::int32_t grid_z __attribute__((aligned(4)));
};
static_assert(sizeof(KktArguments) == 96);

struct __attribute__((packed)) SolveArguments {
  void* ffts __attribute__((aligned(8)));
  void* sync __attribute__((aligned(8)));
  void* workspace __attribute__((aligned(8)));
  void* input __attribute__((aligned(8)));
  void* diagonal __attribute__((aligned(8)));
  void* cumulative __attribute__((aligned(8)));
  void* chunks __attribute__((aligned(8)));
  std::int32_t tokens __attribute__((aligned(4)));
  std::int32_t heads __attribute__((aligned(4)));
  std::int32_t grid_x __attribute__((aligned(4)));
  std::int32_t grid_y __attribute__((aligned(4)));
  std::int32_t grid_z __attribute__((aligned(4)));
};
static_assert(sizeof(SolveArguments) == 80);

struct __attribute__((packed)) MergeArguments {
  void* ffts __attribute__((aligned(8)));
  void* sync __attribute__((aligned(8)));
  void* workspace __attribute__((aligned(8)));
  void* input __attribute__((aligned(8)));
  void* diagonal __attribute__((aligned(8)));
  void* output __attribute__((aligned(8)));
  void* cumulative __attribute__((aligned(8)));
  void* chunks __attribute__((aligned(8)));
  std::int32_t tokens __attribute__((aligned(4)));
  std::int32_t heads __attribute__((aligned(4)));
  std::int32_t grid_x __attribute__((aligned(4)));
  std::int32_t grid_y __attribute__((aligned(4)));
  std::int32_t grid_z __attribute__((aligned(4)));
};
static_assert(sizeof(MergeArguments) == 88);

struct __attribute__((packed)) WyArguments {
  void* ffts __attribute__((aligned(8)));
  void* sync __attribute__((aligned(8)));
  void* workspace __attribute__((aligned(8)));
  void* key __attribute__((aligned(8)));
  void* value __attribute__((aligned(8)));
  void* beta __attribute__((aligned(8)));
  void* w __attribute__((aligned(8)));
  void* u __attribute__((aligned(8)));
  void* inverse __attribute__((aligned(8)));
  void* gate __attribute__((aligned(8)));
  void* cumulative __attribute__((aligned(8)));
  void* chunks __attribute__((aligned(8)));
  std::int32_t tokens __attribute__((aligned(4)));
  std::int32_t heads __attribute__((aligned(4)));
  std::int32_t key_heads __attribute__((aligned(4)));
  std::int32_t key_dim __attribute__((aligned(4)));
  std::int32_t value_dim __attribute__((aligned(4)));
  std::int32_t grid_x __attribute__((aligned(4)));
  std::int32_t grid_y __attribute__((aligned(4)));
  std::int32_t grid_z __attribute__((aligned(4)));
};
static_assert(sizeof(WyArguments) == 128);

struct OwnedError {
  statecentric_qwen38_error_v1 value{};
  std::string message;
};
struct TensorDeleter {
  void operator()(aclTensor* value) const {
    if (value != nullptr) (void)aclDestroyTensor(value);
  }
};
struct IntArrayDeleter {
  void operator()(aclIntArray* value) const {
    if (value != nullptr) (void)aclDestroyIntArray(value);
  }
};
using Tensor = std::unique_ptr<aclTensor, TensorDeleter>;
using IntArray = std::unique_ptr<aclIntArray, IntArrayDeleter>;

int32_t Fail(int32_t status, std::string message,
             statecentric_qwen38_error_v1** out_error) {
  if (out_error != nullptr) {
    auto* owned = new (std::nothrow) OwnedError;
    if (owned != nullptr) {
      owned->message = std::move(message);
      owned->value = {sizeof(statecentric_qwen38_error_v1),
                      STATECENTRIC_QWEN38_PROVIDER_ABI_V1, status, 0,
                      owned->message.c_str(), nullptr};
      *out_error = &owned->value;
    }
  }
  return status;
}

void CheckAcl(aclError status, const char* operation) {
  if (status != ACL_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed with ACL status " +
                             std::to_string(status));
  }
}

void CheckNn(aclnnStatus status, const char* operation) {
  if (status != 0) {
    throw std::runtime_error(std::string(operation) +
                             " failed with ACLNN status " +
                             std::to_string(status));
  }
}

void CheckRt(rtError_t status, const char* operation) {
  if (status != RT_ERROR_NONE) {
    throw std::runtime_error(std::string(operation) + " failed with RT status " +
                             std::to_string(status));
  }
}

const char* RequiredEnvironment(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    throw std::runtime_error(std::string("missing GDN artifact path: ") + name);
  }
  return value;
}

Tensor MakeTensor(const std::vector<std::int64_t>& shape,
                  const std::vector<std::int64_t>& strides,
                  const std::vector<std::int64_t>& storage_shape,
                  aclDataType type, void* data,
                  std::int64_t storage_offset = 0) {
  auto* tensor = aclCreateTensor(shape.data(), shape.size(), type,
                                 strides.data(), storage_offset, ACL_FORMAT_ND,
                                 storage_shape.data(), storage_shape.size(), data);
  if (tensor == nullptr) throw std::runtime_error("aclCreateTensor(GDN) failed");
  return Tensor(tensor);
}

void WriteBestEffort(int descriptor, const char* data, std::size_t bytes) {
  const ssize_t written = ::write(descriptor, data, bytes);
  (void)written;
}

#ifdef STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1
void* AllocatePrivate(statecentric_qwen38_event_v1* event, std::size_t bytes) {
  if (bytes == 0) return nullptr;
  // Allocate metadata first, so a host allocation failure cannot orphan NPU memory.
  event->allocations.reserve(event->allocations.size() + 1);
  void* value = nullptr;
  CheckAcl(aclrtMalloc(&value, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
           "aclrtMalloc(GDN private tensor/workspace)");
  event->allocations.push_back(value);
  return value;
}
#endif

#ifdef STATECENTRIC_DEVICE_ALLOCATION_CATEGORIES
void* Allocate(statecentric_qwen38_event_v1* event, std::size_t bytes,
    statecentric::DeviceAllocationCategory category = statecentric::DeviceAllocationCategory::kTensor) {
#else
void* Allocate(statecentric_qwen38_event_v1* event, std::size_t bytes) {
#endif
  if (bytes == 0) return nullptr;
#ifdef STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1
  if (event->tensor_pool != nullptr) {
    event->tensor_leases.reserve(event->tensor_leases.size() + 1);
    auto lease = event->tensor_pool->Borrow(
        reinterpret_cast<std::uintptr_t>(event), bytes);
    if (lease) {
      event->tensor_leases.push_back(*lease);
      return lease->pointer;
    }
  }
  return AllocatePrivate(event, bytes);
#else
#ifdef STATECENTRIC_DEVICE_ALLOCATION_DIAGNOSTICS
  void* value = event_allocation_ledger.Allocate(bytes,
#ifdef STATECENTRIC_DEVICE_ALLOCATION_CATEGORIES
      static_cast<std::size_t>(category),
#endif
      [](std::size_t size) {
    void* pointer = nullptr;
    CheckAcl(aclrtMalloc(&pointer, size, ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(GDN provider)");
    return pointer;
  });
  try {
    event->allocations.push_back(value);
  } catch (...) {
    event_allocation_ledger.Free(value, [](void* pointer) {
      return aclrtFree(pointer) == ACL_SUCCESS;
    });
    throw;
  }
#else
  void* value = nullptr;
  CheckAcl(aclrtMalloc(&value, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
           "aclrtMalloc(GDN provider)");
  event->allocations.push_back(value);
#endif
  return value;
#endif
}

Tensor CopyInt64MetadataTensor(statecentric_qwen38_event_v1* event,
                               std::span<const std::int64_t> values,
                               const char* operation) {
  if (values.empty()) {
    throw std::runtime_error(std::string(operation) + " cannot be empty");
  }
  const auto bytes = values.size_bytes();
  void* device = Allocate(event, bytes);
  CheckAcl(aclrtMemcpy(device, bytes, values.data(), bytes,
                       ACL_MEMCPY_HOST_TO_DEVICE),
           operation);
  const auto elements = static_cast<std::int64_t>(values.size());
  return MakeTensor({elements}, {1}, {elements}, ACL_INT64, device);
}

void FreeAllocations(statecentric_qwen38_event_v1* event) {
#ifdef STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1
  for (const auto& lease : event->tensor_leases) {
    event->tensor_pool->Retire(lease, event->complete);
  }
  event->tensor_leases.clear();
#endif
  for (auto it = event->allocations.rbegin(); it != event->allocations.rend();
       ++it) {
#ifdef STATECENTRIC_DEVICE_ALLOCATION_DIAGNOSTICS
    event_allocation_ledger.Free(*it, [](void* value) {
      return aclrtFree(value) == ACL_SUCCESS;
    });
#else
    (void)aclrtFree(*it);
#endif
  }
  event->allocations.clear();
}

void ReleaseSplitCaches(statecentric_qwen38_provider_handle_v1* handle,
                        statecentric_qwen38_event_v1* event) {
  if (handle == nullptr || event == nullptr || event->split_caches.empty()) {
    return;
  }
  std::scoped_lock lock(handle->split_cache_mutex);
  for (auto* cache : event->split_caches) {
    if (cache != nullptr) cache->in_use = false;
  }
  event->split_caches.clear();
}

void DestroySynchronizationEvents(statecentric_qwen38_event_v1* event) {
  if (event == nullptr) return;
  for (auto it = event->synchronization_events.rbegin();
       it != event->synchronization_events.rend(); ++it) {
    if (*it != nullptr) (void)aclrtDestroyEvent(*it);
  }
  event->synchronization_events.clear();
}

void EmitLaneAnchoredFirstAdmissionV14(
    statecentric_qwen38_provider_handle_v1* handle) {
  if (handle == nullptr) return;
  std::array<char, 256> line{};
  const int length = std::snprintf(
      line.data(), line.size(),
      "STATECENTRIC_QWEN38_GDN_LANE_ANCHORED_TELEMETRY_V14="
      "{\"rank\":%u,\"submit_calls\":%llu,\"admitted_calls\":%llu}\n",
      handle->rank,
      static_cast<unsigned long long>(
          handle->lane_anchored_submit_calls_v14.load(
              std::memory_order_relaxed)),
      static_cast<unsigned long long>(
          handle->lane_anchored_admitted_calls_v14.load(
              std::memory_order_relaxed)));
  if (length > 0 && static_cast<std::size_t>(length) < line.size()) {
    WriteBestEffort(STDERR_FILENO, line.data(),
                    static_cast<std::size_t>(length));
  }
}

void EmitPackedDecodeFirstAdmissionV12(
    statecentric_qwen38_provider_handle_v1* handle) {
  if (handle == nullptr) return;
  std::array<char, 256> line{};
  const int length = std::snprintf(
      line.data(), line.size(),
      "STATECENTRIC_QWEN38_GDN_PACKED_DECODE_TELEMETRY_V12="
      "{\"rank\":%u,\"submit_calls\":%llu,\"admitted_calls\":%llu}\n",
      handle->rank,
      static_cast<unsigned long long>(
          handle->packed_decode_submit_calls_v12.load(
              std::memory_order_relaxed)),
      static_cast<unsigned long long>(
          handle->packed_decode_admitted_calls_v12.load(
              std::memory_order_relaxed)));
  if (length > 0 && static_cast<std::size_t>(length) < line.size()) {
    WriteBestEffort(STDERR_FILENO, line.data(),
                    static_cast<std::size_t>(length));
  }
}

void EmitGraphCapturedLaneFirstAdmissionV15(
    statecentric_qwen38_provider_handle_v1* handle) {
  if (handle == nullptr) return;
  std::array<char, 256> line{};
  const int length = std::snprintf(
      line.data(), line.size(),
      "STATECENTRIC_QWEN38_GDN_GRAPH_CAPTURE_EX_SYNC_V15="
      "{\"rank\":%u,\"captured_admissions\":%llu}\n",
      handle->rank,
      static_cast<unsigned long long>(
          handle->graph_captured_lane_admissions_v15.load(
              std::memory_order_relaxed)));
  if (length > 0 && static_cast<std::size_t>(length) < line.size()) {
    WriteBestEffort(STDERR_FILENO, line.data(),
                    static_cast<std::size_t>(length));
  }
}

void EmitGraphSingleFollowerFirstAdmissionV16(
    statecentric_qwen38_provider_handle_v1* handle) {
  if (handle == nullptr) return;
  std::array<char, 256> line{};
  const int length = std::snprintf(
      line.data(), line.size(),
      "STATECENTRIC_QWEN38_GDN_GRAPH_SINGLE_FOLLOWER_V16="
      "{\"rank\":%u,\"captured_admissions\":%llu,\"events_per_layer\":4,"
      "\"waits_per_layer\":4}\n",
      handle->rank,
      static_cast<unsigned long long>(
          handle->graph_single_follower_admissions_v16.load(
              std::memory_order_relaxed)));
  if (length > 0 && static_cast<std::size_t>(length) < line.size()) {
    WriteBestEffort(STDERR_FILENO, line.data(),
                    static_cast<std::size_t>(length));
  }
}

void EmitEagerSingleFollowerFirstAdmissionV17(
    statecentric_qwen38_provider_handle_v1* handle) {
  if (handle == nullptr) return;
  std::array<char, 256> line{};
  const int length = std::snprintf(
      line.data(), line.size(),
      "STATECENTRIC_QWEN38_GDN_EAGER_SINGLE_FOLLOWER_V17="
      "{\"rank\":%u,\"eager_admissions\":%llu,\"events_per_layer\":4,"
      "\"waits_per_layer\":4}\n",
      handle->rank,
      static_cast<unsigned long long>(
          handle->eager_single_follower_admissions_v17.load(
              std::memory_order_relaxed)));
  if (length > 0 && static_cast<std::size_t>(length) < line.size()) {
    WriteBestEffort(STDERR_FILENO, line.data(),
                    static_cast<std::size_t>(length));
  }
}

void EmitDecodeFlatOutputFirstAdmissionV18(
    statecentric_qwen38_provider_handle_v1* handle) {
  if (handle == nullptr) return;
  std::array<char, 256> line{};
  const int length = std::snprintf(
      line.data(), line.size(),
      "STATECENTRIC_QWEN38_GDN_DECODE_FLAT_OUTPUT_PROJECTION_V18="
      "{\"rank\":%u,\"admitted_calls\":%llu,\"rows\":2}\n",
      handle->rank,
      static_cast<unsigned long long>(
          handle->decode_flat_output_admissions_v18.load(
              std::memory_order_relaxed)));
  if (length > 0 && static_cast<std::size_t>(length) < line.size()) {
    WriteBestEffort(STDERR_FILENO, line.data(),
                    static_cast<std::size_t>(length));
  }
}

void EmitParentLocalDecodeTailFirstAdmissionV19(
    statecentric_qwen38_provider_handle_v1* handle) {
  if (handle == nullptr) return;
  std::array<char, 256> line{};
  const int length = std::snprintf(
      line.data(), line.size(),
      "STATECENTRIC_QWEN38_GDN_PARENT_LOCAL_DECODE_TAIL_V19="
      "{\"rank\":%u,\"admitted_calls\":%llu,\"rows\":2,"
      "\"record_wait_edges_removed\":2}\n",
      handle->rank,
      static_cast<unsigned long long>(
          handle->parent_local_decode_tail_admissions_v19.load(
              std::memory_order_relaxed)));
  if (length > 0 && static_cast<std::size_t>(length) < line.size()) {
    WriteBestEffort(STDERR_FILENO, line.data(),
                    static_cast<std::size_t>(length));
  }
}

void EmitEventPoolRuntimeTelemetryLocked(
    statecentric_qwen38_provider_handle_v1* handle, const char* reason) {
  if (handle == nullptr ||
      (!handle->event_pool_runtime_telemetry_v7 &&
       !handle->event_pool_atomic_runtime_telemetry_v8)) {
    return;
  }
  const auto active_in_use =
      handle->synchronization_event_pool_acquisitions -
      handle->synchronization_event_pool_releases;
  const char* prefix = handle->event_pool_atomic_runtime_telemetry_v8
                           ? "STATECENTRIC_QWEN38_GDN_EVENT_POOL_TELEMETRY_V8="
                           : "STATECENTRIC_QWEN38_GDN_EVENT_POOL_TELEMETRY_V7=";
  std::array<char, 512> line{};
  const int length = std::snprintf(
      line.data(), line.size(),
      "%s{\"rank\":%u,"
      "\"reason\":\"%s\",\"acquire_attempts\":%llu,"
      "\"acquisitions\":%llu,\"fallbacks\":%llu,\"releases\":%llu,"
      "\"active_in_use\":%llu,\"peak_in_use\":%zu,"
      "\"active_bundle_count\":%zu,"
      "\"bundle_leases\":[%llu,%llu,%llu,%llu]}\n",
      prefix, handle->rank, reason,
      static_cast<unsigned long long>(
          handle->synchronization_event_pool_acquire_attempts),
      static_cast<unsigned long long>(
          handle->synchronization_event_pool_acquisitions),
      static_cast<unsigned long long>(
          handle->synchronization_event_pool_fallbacks),
      static_cast<unsigned long long>(
          handle->synchronization_event_pool_releases),
      static_cast<unsigned long long>(active_in_use),
      handle->synchronization_event_pool_peak_in_use,
      handle->synchronization_event_pool_active_bundles_v9,
      static_cast<unsigned long long>(
          handle->synchronization_event_pool_bundle_leases[0]),
      static_cast<unsigned long long>(
          handle->synchronization_event_pool_bundle_leases[1]),
      static_cast<unsigned long long>(
          handle->synchronization_event_pool_bundle_leases[2]),
      static_cast<unsigned long long>(
          handle->synchronization_event_pool_bundle_leases[3]));
  if (length <= 0 || static_cast<std::size_t>(length) >= line.size()) return;
  if (handle->event_pool_atomic_runtime_telemetry_v8) {
    WriteBestEffort(STDERR_FILENO, line.data(),
                    static_cast<std::size_t>(length));
  } else {
    (void)std::fwrite(line.data(), 1, static_cast<std::size_t>(length), stderr);
    std::fflush(stderr);
  }
}

statecentric_qwen38_synchronization_event_bundle_v1*
AcquireSynchronizationEventBundle(
    statecentric_qwen38_provider_handle_v1* handle) {
  if (handle == nullptr) return nullptr;
  std::scoped_lock lock(handle->synchronization_event_pool_mutex);
  ++handle->synchronization_event_pool_acquire_attempts;
  std::array<bool, statecentric::kQwen38SynchronizationEventBundleCount>
      in_use{};
  std::transform(handle->synchronization_event_bundles.begin(),
                 handle->synchronization_event_bundles.end(), in_use.begin(),
                 [](const auto& bundle) { return bundle.in_use; });
  const auto selected = statecentric::SelectQwen38SynchronizationEventBundle(
      in_use, handle->synchronization_event_pool_cursor,
      handle->synchronization_event_pool_active_bundles_v9);
  if (!selected.has_value()) {
    ++handle->synchronization_event_pool_fallbacks;
    EmitEventPoolRuntimeTelemetryLocked(handle, "fallback");
    return nullptr;
  }
  auto& bundle = handle->synchronization_event_bundles[*selected];
  std::vector<std::size_t> created;
  try {
    for (std::size_t index = 0; index < bundle.events.size(); ++index) {
      if (bundle.events[index] != nullptr) continue;
      if (handle->reusable_event_ex_v5) {
        CheckAcl(aclrtCreateEventExWithFlag(&bundle.events[index],
                                            ACL_EVENT_SYNC),
                 "aclrtCreateEventExWithFlag(GDN reusable synchronization event)");
      } else {
        CheckAcl(aclrtCreateEvent(&bundle.events[index]),
                 "aclrtCreateEvent(GDN pooled synchronization event)");
      }
      created.push_back(index);
    }
  } catch (...) {
    for (auto it = created.rbegin(); it != created.rend(); ++it) {
      (void)aclrtDestroyEvent(bundle.events[*it]);
      bundle.events[*it] = nullptr;
    }
    throw;
  }
  bundle.in_use = true;
  ++handle->synchronization_event_pool_acquisitions;
  ++handle->synchronization_event_pool_bundle_leases[*selected];
  const auto previous_peak = handle->synchronization_event_pool_peak_in_use;
  handle->synchronization_event_pool_peak_in_use = std::max(
      handle->synchronization_event_pool_peak_in_use,
      static_cast<std::size_t>(std::count(in_use.begin(), in_use.end(), true)) +
          1);
  if (handle->synchronization_event_pool_peak_in_use > previous_peak) {
    EmitEventPoolRuntimeTelemetryLocked(handle, "peak");
  }
  handle->synchronization_event_pool_cursor =
      (*selected + 1) % handle->synchronization_event_pool_active_bundles_v9;
  return &bundle;
}

void ReleaseSynchronizationEventBundle(
    statecentric_qwen38_provider_handle_v1* handle,
    statecentric_qwen38_event_v1* event) {
  if (handle == nullptr || event == nullptr ||
      event->synchronization_event_bundle == nullptr) {
    return;
  }
  std::scoped_lock lock(handle->synchronization_event_pool_mutex);
  event->synchronization_event_bundle->in_use = false;
  ++handle->synchronization_event_pool_releases;
  if (handle->synchronization_event_pool_releases == 1 ||
      handle->synchronization_event_pool_releases % 64 == 0) {
    EmitEventPoolRuntimeTelemetryLocked(handle, "periodic-release");
  }
  event->synchronization_event_bundle = nullptr;
}

aclrtEvent SynchronizationEvent(
    statecentric_qwen38_event_v1* aggregate, std::size_t index,
    const char* label) {
  if (aggregate == nullptr || index >= kQwen38SynchronizationEventsPerBundle) {
    throw std::runtime_error("invalid GDN synchronization event slot");
  }
  if (aggregate->synchronization_event_bundle != nullptr) {
    const auto event = aggregate->synchronization_event_bundle->events[index];
    if (event == nullptr) {
      throw std::runtime_error("missing pooled GDN synchronization event");
    }
    return event;
  }
  aclrtEvent event = nullptr;
  if (aggregate->model_ri_cross_stream_capture) {
    CheckAcl(aclrtCreateEventExWithFlag(&event, ACL_EVENT_SYNC), label);
  } else {
    CheckAcl(aclrtCreateEvent(&event), label);
  }
  aggregate->synchronization_events.push_back(event);
  return event;
}

bool GraphCaptureActive(aclrtStream stream) {
  if (stream == nullptr) {
    throw std::invalid_argument("missing GDN stream for model-RI capture query");
  }
  aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI model = nullptr;
  CheckAcl(aclmdlRICaptureGetInfo(stream, &status, &model),
           "aclmdlRICaptureGetInfo(GDN cross-stream events)");
  if (status == ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED) {
    throw std::runtime_error("GDN observed invalidated model-RI capture");
  }
  return status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
}

#ifdef STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1
void BindPrefillTensorPool(statecentric_qwen38_provider_handle_v1* handle,
                          statecentric_qwen38_event_v1* event,
                          std::uint32_t tokens, bool has_precomputed) {
  // Parent/child buffers and graph storage have different consumer lifetimes.
  const bool sequence_child = std::find(handle->sequence_streams.begin(),
      handle->sequence_streams.end(), event->stream) != handle->sequence_streams.end();
  if (tokens < kChunkKernelMinimumTokens || has_precomputed || sequence_child) return;
  // Require NONE explicitly; invalidated or unknown capture states cannot admit.
  aclmdlRICaptureStatus status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
  aclmdlRI model = nullptr;
  CheckAcl(aclmdlRICaptureGetInfo(event->stream, &status, &model),
           "aclmdlRICaptureGetInfo(GDN prefill tensor pool v1)");
  if (status == ACL_MODEL_RI_CAPTURE_STATUS_NONE) {
    event->tensor_pool = handle->prefill_tensor_pool.get();
  }
}
#endif

void ReleaseSequenceAggregate(statecentric_qwen38_provider_handle_v1* handle,
                              statecentric_qwen38_event_v1* aggregate,
                              bool synchronize_streams) {
  if (aggregate == nullptr) return;
  if (synchronize_streams) {
    (void)aclrtSynchronizeStream(aggregate->stream);
    if (handle != nullptr) {
      for (const auto stream : handle->sequence_streams) {
        if (stream != nullptr) (void)aclrtSynchronizeStream(stream);
      }
    }
  }
  for (auto* child : aggregate->sequence_events) {
    if (child == nullptr) continue;
    child->complete = true;
    ReleaseSplitCaches(handle, child);
    FreeAllocations(child);
    DestroySynchronizationEvents(child);
    delete child;
  }
  aggregate->sequence_events.clear();
  DestroySynchronizationEvents(aggregate);
  ReleaseSynchronizationEventBundle(handle, aggregate);
  ReleaseSplitCaches(handle, aggregate);
  FreeAllocations(aggregate);
  delete aggregate;
}

void FreeHandleConstants(statecentric_qwen38_provider_handle_v1* handle) {
  if (handle == nullptr) return;
  {
    std::scoped_lock lock(handle->split_cache_mutex);
    for (auto& caches : handle->split_caches) caches.clear();
  }
  for (auto& stream : handle->sequence_streams) {
    if (stream != nullptr) {
      (void)aclrtSynchronizeStream(stream);
      (void)aclrtDestroyStream(stream);
    }
    stream = nullptr;
  }
  {
    std::scoped_lock lock(handle->synchronization_event_pool_mutex);
    EmitEventPoolRuntimeTelemetryLocked(handle, "destruction");
    if (handle->event_pool_telemetry_v6) {
      std::fprintf(
          stderr,
          "STATECENTRIC_QWEN38_GDN_EVENT_POOL_TELEMETRY_V6={\"rank\":%u,"
          "\"acquire_attempts\":%llu,\"acquisitions\":%llu,"
          "\"fallbacks\":%llu,\"releases\":%llu,\"peak_in_use\":%zu,"
          "\"bundle_leases\":[%llu,%llu,%llu,%llu]}\n",
          handle->rank,
          static_cast<unsigned long long>(
              handle->synchronization_event_pool_acquire_attempts),
          static_cast<unsigned long long>(
              handle->synchronization_event_pool_acquisitions),
          static_cast<unsigned long long>(
              handle->synchronization_event_pool_fallbacks),
          static_cast<unsigned long long>(
              handle->synchronization_event_pool_releases),
          handle->synchronization_event_pool_peak_in_use,
          static_cast<unsigned long long>(
              handle->synchronization_event_pool_bundle_leases[0]),
          static_cast<unsigned long long>(
              handle->synchronization_event_pool_bundle_leases[1]),
          static_cast<unsigned long long>(
              handle->synchronization_event_pool_bundle_leases[2]),
          static_cast<unsigned long long>(
              handle->synchronization_event_pool_bundle_leases[3]));
      std::fflush(stderr);
    }
    for (auto& bundle : handle->synchronization_event_bundles) {
      for (auto& event : bundle.events) {
        if (event != nullptr) (void)aclrtDestroyEvent(event);
        event = nullptr;
      }
      bundle.in_use = false;
    }
    handle->synchronization_event_pool_cursor = 0;
  }
  handle->wy_kernel.reset();
  handle->merge_kernel.reset();
  handle->solve_kernel.reset();
  handle->kkt_kernel.reset();
  handle->cumsum_kernel.reset();
  if (handle->chunk_indices != nullptr) (void)aclrtFree(handle->chunk_indices);
  handle->chunk_indices = nullptr;
  for (void*& value : handle->state_indices) {
    if (value != nullptr) (void)aclrtFree(value);
    value = nullptr;
  }
  for (void*& value : handle->actual_lengths) {
    if (value != nullptr) (void)aclrtFree(value);
    value = nullptr;
  }
  if (handle->packed_decode_actual_lengths != nullptr) {
    (void)aclrtFree(handle->packed_decode_actual_lengths);
  }
  handle->packed_decode_actual_lengths = nullptr;
  if (handle->packed_decode_state_indices != nullptr) {
    (void)aclrtFree(handle->packed_decode_state_indices);
  }
  handle->packed_decode_state_indices = nullptr;
  if (handle->l2_gamma != nullptr) (void)aclrtFree(handle->l2_gamma);
  handle->l2_gamma = nullptr;
}

void EnsureSequenceStreams(statecentric_qwen38_provider_handle_v1* handle) {
  if (handle == nullptr) {
    throw std::runtime_error("missing GDN handle for sequence streams");
  }
  std::scoped_lock lock(handle->sequence_stream_mutex);
  if (std::all_of(handle->sequence_streams.begin(),
                  handle->sequence_streams.end(),
                  [](aclrtStream stream) { return stream != nullptr; })) {
    return;
  }
  std::vector<std::size_t> created;
  try {
    for (std::size_t lane = 0; lane < handle->sequence_streams.size(); ++lane) {
      auto& stream = handle->sequence_streams[lane];
      if (stream != nullptr) continue;
      CheckAcl(aclrtCreateStream(&stream),
               "aclrtCreateStream(GDN sequence lane)");
      created.push_back(lane);
    }
  } catch (...) {
    for (auto it = created.rbegin(); it != created.rend(); ++it) {
      auto& stream = handle->sequence_streams[*it];
      if (stream != nullptr) (void)aclrtDestroyStream(stream);
      stream = nullptr;
    }
    throw;
  }
}

void EnsureHandleConstants(statecentric_qwen38_provider_handle_v1* handle) {
  if (handle->l2_gamma != nullptr) return;
  try {
    std::array<std::uint16_t, kHeadDim> l2_gamma{};
    const float l2_value = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    const std::uint32_t l2_bits = std::bit_cast<std::uint32_t>(l2_value);
    const std::uint16_t l2_bf16 = static_cast<std::uint16_t>(
        (l2_bits + 0x7FFFU + ((l2_bits >> 16U) & 1U)) >> 16U);
    l2_gamma.fill(l2_bf16);
    CheckAcl(aclrtMalloc(&handle->l2_gamma, sizeof(l2_gamma),
                         ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(GDN resident L2 gamma)");
    CheckAcl(aclrtMemcpy(handle->l2_gamma, sizeof(l2_gamma), l2_gamma.data(),
                         sizeof(l2_gamma), ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(GDN resident L2 gamma)");
    for (std::uint32_t token_count = 1;
         token_count <= kMaximumSubmissionTokens; ++token_count) {
      const std::array<std::int32_t, 2> lengths{
          0, static_cast<std::int32_t>(token_count)};
      const std::vector<std::int32_t> indices(token_count, 0);
      const auto index = static_cast<std::size_t>(token_count - 1);
      CheckAcl(aclrtMalloc(&handle->actual_lengths[index], sizeof(lengths),
                           ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc(GDN resident actual lengths)");
      CheckAcl(aclrtMemcpy(handle->actual_lengths[index], sizeof(lengths),
                           lengths.data(), sizeof(lengths),
                           ACL_MEMCPY_HOST_TO_DEVICE),
               "aclrtMemcpy(GDN resident actual lengths)");
      if (token_count <= kRecurrentKernelTokens) {
        CheckAcl(aclrtMalloc(&handle->state_indices[index],
                             indices.size() * sizeof(std::int32_t),
                             ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(GDN resident state indices)");
        CheckAcl(aclrtMemcpy(handle->state_indices[index],
                             indices.size() * sizeof(std::int32_t),
                             indices.data(),
                             indices.size() * sizeof(std::int32_t),
                             ACL_MEMCPY_HOST_TO_DEVICE),
                 "aclrtMemcpy(GDN resident state indices)");
      }
    }
    const std::array<std::int32_t, 2> chunk_indices{0, 0};
    CheckAcl(aclrtMalloc(&handle->chunk_indices, sizeof(chunk_indices),
                         ACL_MEM_MALLOC_HUGE_FIRST),
             "aclrtMalloc(GDN resident chunk indices)");
    CheckAcl(aclrtMemcpy(handle->chunk_indices, sizeof(chunk_indices),
                         chunk_indices.data(), sizeof(chunk_indices),
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(GDN resident chunk indices)");
    if (statecentric::Qwen38GdnPackedDecodeConstantsRequired(
            handle->packed_decode_recurrent_v12,
            handle->lane_local_packed_recurrent_v13,
            handle->lane_anchored_packed_recurrent_v14)) {
      const std::array<std::int32_t, 3> packed_lengths{0, 1, 1};
      const std::array<std::int32_t, 2> packed_indices{0, 1};
      CheckAcl(aclrtMalloc(&handle->packed_decode_actual_lengths,
                           sizeof(packed_lengths), ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc(GDN packed decode actual lengths v12)");
      CheckAcl(aclrtMemcpy(handle->packed_decode_actual_lengths,
                           sizeof(packed_lengths), packed_lengths.data(),
                           sizeof(packed_lengths), ACL_MEMCPY_HOST_TO_DEVICE),
               "aclrtMemcpy(GDN packed decode actual lengths v12)");
      CheckAcl(aclrtMalloc(&handle->packed_decode_state_indices,
                           sizeof(packed_indices), ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc(GDN packed decode state indices v12)");
      CheckAcl(aclrtMemcpy(handle->packed_decode_state_indices,
                           sizeof(packed_indices), packed_indices.data(),
                           sizeof(packed_indices), ACL_MEMCPY_HOST_TO_DEVICE),
               "aclrtMemcpy(GDN packed decode state indices v12)");
    }
  } catch (...) {
    FreeHandleConstants(handle);
    throw;
  }
}

void InitializeChunkKernels(statecentric_qwen38_provider_handle_v1* handle,
                            std::int32_t device) {
  CheckRt(rtGetAiCoreCount(&handle->ai_cores), "rtGetAiCoreCount(GDN chunk)");
  if (handle->ai_cores != 24) {
    throw std::runtime_error("GDN chunk provider requires 24 AI cores");
  }
  std::uint32_t ffts_bytes = 0;
  CheckRt(rtGetC2cCtrlAddr(reinterpret_cast<std::uint64_t*>(&handle->ffts),
                           &ffts_bytes),
          "rtGetC2cCtrlAddr(GDN chunk)");
  if (handle->ffts == nullptr || ffts_bytes != 32) {
    throw std::runtime_error("GDN chunk FFTS control contract mismatch");
  }
  const bool tp2 = handle->query_heads == 8 && handle->value_heads == 24;
  const char* cumsum_sha = tp2 ? kTp2CumsumSha256 : kTp4CumsumSha256;
  const char* kkt_sha = tp2 ? kTp2KktSha256 : kTp4KktSha256;
  const char* solve_sha = tp2 ? kTp2SolveSha256 : kTp4SolveSha256;
  const char* merge_sha = tp2 ? kTp2MergeSha256 : kTp4MergeSha256;
  const char* wy_sha = handle->wy_outputs_bhtd
                           ? kTp2WyBhtdSha256
                           : (tp2 ? kTp2WySha256 : kTp4WySha256);
  handle->cumsum_kernel =
      std::make_unique<statecentric::Qwen38TritonAotKernel>(
          RequiredEnvironment("STATECENTRIC_QWEN38_GDN_CUMSUM_AOT"),
          cumsum_sha, "chunk_local_cumsum_scalar_kernel",
          statecentric::Qwen38TritonKernelMode::kAiv, device);
  handle->kkt_kernel =
      std::make_unique<statecentric::Qwen38TritonAotKernel>(
          RequiredEnvironment("STATECENTRIC_QWEN38_GDN_KKT_AOT"), kkt_sha,
          "chunk_scaled_dot_kkt_fwd_kernel",
          statecentric::Qwen38TritonKernelMode::kMix, device);
  handle->solve_kernel =
      std::make_unique<statecentric::Qwen38TritonAotKernel>(
          RequiredEnvironment("STATECENTRIC_QWEN38_GDN_SOLVE_AOT"),
          solve_sha, "solve_tril_16x16_kernel",
          statecentric::Qwen38TritonKernelMode::kAiv, device);
  handle->merge_kernel =
      std::make_unique<statecentric::Qwen38TritonAotKernel>(
          RequiredEnvironment("STATECENTRIC_QWEN38_GDN_MERGE_AOT"),
          merge_sha, "merge_16x16_to_64x64_inverse_kernel",
          statecentric::Qwen38TritonKernelMode::kMix, device);
  handle->wy_kernel =
      std::make_unique<statecentric::Qwen38TritonAotKernel>(
          RequiredEnvironment(handle->wy_outputs_bhtd
                                  ? "STATECENTRIC_QWEN38_GDN_WY_BHTD_AOT"
                                  : "STATECENTRIC_QWEN38_GDN_WY_AOT"),
          wy_sha,
          handle->wy_outputs_bhtd ? "recompute_w_u_fwd_bhtd_kernel"
                                  : "recompute_w_u_fwd_kernel",
          statecentric::Qwen38TritonKernelMode::kMix, device);
}

void InitializePostConvPackKernel(
    statecentric_qwen38_provider_handle_v1* handle, std::int32_t device) {
  if (handle->query_heads != 8 || handle->value_heads != 24 ||
      handle->qkv != 5120 || handle->value != 3072) {
    throw std::runtime_error(
        "GDN post-conv pack provider requires the TP2 geometry");
  }
  handle->post_conv_pack_kernel =
      std::make_unique<statecentric::Qwen38TritonAotKernel>(
          RequiredEnvironment("STATECENTRIC_QWEN38_GDN_POST_CONV_PACK_AOT"),
          kTp2PostConvPackSha256, "gdn_post_conv_pack_kernel",
          statecentric::Qwen38TritonKernelMode::kAiv, device);
}

void PostConvPack(statecentric_qwen38_provider_handle_v1* handle,
                  std::uint32_t tokens, void* mixed_qkv, void* query,
                  void* key, void* value, aclrtStream stream) {
  if (handle == nullptr || handle->post_conv_pack_kernel == nullptr ||
      handle->ffts == nullptr || tokens == 0 ||
      tokens > kMaximumSubmissionTokens || mixed_qkv == nullptr ||
      query == nullptr || key == nullptr || value == nullptr ||
      stream == nullptr) {
    throw std::runtime_error("invalid GDN post-conv pack launch");
  }
  constexpr std::int32_t kMixedQkvTokenStride = 5120;
  constexpr std::int32_t kQueryTokenStride = 1024;
  constexpr std::int32_t kValueTokenStride = 3072;
  constexpr std::uint32_t kHeads = 8 + 24;
  constexpr std::uint32_t kBlockTokens = 16;
  const auto blocks = ((tokens + kBlockTokens - 1) / kBlockTokens) * kHeads;
  PostConvPackArguments arguments{
      handle->ffts,
      nullptr,
      nullptr,
      mixed_qkv,
      query,
      key,
      value,
      kMixedQkvTokenStride,
      kQueryTokenStride,
      kQueryTokenStride,
      kValueTokenStride,
      static_cast<std::int32_t>(tokens),
      static_cast<std::int32_t>(blocks),
      1,
      1,
  };
  handle->post_conv_pack_kernel->Launch(blocks, &arguments, sizeof(arguments),
                                        stream);
}

void Matmul(const Tensor& input, const Tensor& weight, const Tensor& output,
            aclrtStream stream, statecentric_qwen38_event_v1* event,
            const char* label) {
  std::uint64_t bytes = 0;
  aclOpExecutor* executor = nullptr;
  CheckNn(aclnnMatmulGetWorkspaceSize(input.get(), weight.get(), output.get(),
                                      0, &bytes, &executor), label);
  void* workspace = GDN_WORKSPACE(event, bytes, kMatmulWorkspace);
  CheckNn(aclnnMatmul(workspace, bytes, executor, stream), label);
}

void BatchMatmulSharedWeight(void* input_data, void* weight_data,
                             void* output_data, std::int64_t batch,
                             std::int64_t tokens, std::int64_t input_columns,
                             std::int64_t output_columns, aclrtStream stream,
                             statecentric_qwen38_event_v1* event,
                             const char* label) {
  auto input = MakeTensor(
      {batch, tokens, input_columns},
      {tokens * input_columns, input_columns, 1},
      {batch, tokens, input_columns}, ACL_BF16, input_data);
  auto weight = MakeTensor(
      {1, input_columns, output_columns},
      {input_columns * output_columns, 1, input_columns},
      {1, output_columns, input_columns}, ACL_BF16, weight_data);
  auto output = MakeTensor(
      {batch, tokens, output_columns},
      {tokens * output_columns, output_columns, 1},
      {batch, tokens, output_columns}, ACL_BF16, output_data);
  std::uint64_t bytes = 0;
  aclOpExecutor* executor = nullptr;
  CheckNn(aclnnBatchMatMulGetWorkspaceSize(input.get(), weight.get(),
                                            output.get(), 0, &bytes, &executor),
          label);
  void* workspace = GDN_WORKSPACE(event, bytes, kMatmulWorkspace);
  CheckNn(aclnnBatchMatMul(workspace, bytes, executor, stream), label);
}

void FlatMatmulSharedWeight(void* input_data, void* weight_data,
                            void* output_data, std::int64_t rows,
                            std::int64_t input_columns,
                            std::int64_t output_columns, aclrtStream stream,
                            statecentric_qwen38_event_v1* event,
                            const char* label) {
  auto input = MakeTensor({rows, input_columns}, {input_columns, 1},
                          {rows, input_columns}, ACL_BF16, input_data);
  auto weight = MakeTensor(
      {input_columns, output_columns}, {1, input_columns},
      {output_columns, input_columns}, ACL_BF16, weight_data);
  auto output = MakeTensor({rows, output_columns}, {output_columns, 1},
                           {rows, output_columns}, ACL_BF16, output_data);
  Matmul(input, weight, output, stream, event, label);
}

void RmsNorm(const Tensor& input, const Tensor& gamma, const Tensor& output,
             std::int64_t rows, double epsilon, aclrtStream stream,
             statecentric_qwen38_event_v1* event, const char* label) {
  void* rstd = Allocate(event, static_cast<std::size_t>(rows) * sizeof(float));
  auto rstd_tensor = MakeTensor({rows, 1}, {1, 1}, {rows, 1}, ACL_FLOAT, rstd);
  std::uint64_t bytes = 0;
  aclOpExecutor* executor = nullptr;
  CheckNn(aclnnRmsNormGetWorkspaceSize(input.get(), gamma.get(), epsilon,
                                       output.get(), rstd_tensor.get(), &bytes,
                                       &executor), label);
  void* workspace = GDN_WORKSPACE(event, bytes, kNormalizationWorkspace);
  CheckNn(aclnnRmsNorm(workspace, bytes, executor, stream), label);
}

void RmsNormHeads(const Tensor& input, const Tensor& gamma,
                  const Tensor& output, std::int64_t tokens,
                  std::int64_t heads, double epsilon, aclrtStream stream,
                  statecentric_qwen38_event_v1* event, const char* label) {
  void* rstd = Allocate(
      event, static_cast<std::size_t>(tokens * heads) * sizeof(float));
  auto rstd_tensor = MakeTensor({tokens, heads, 1}, {heads, 1, 1},
                                {tokens, heads, 1}, ACL_FLOAT, rstd);
  std::uint64_t bytes = 0;
  aclOpExecutor* executor = nullptr;
  CheckNn(aclnnRmsNormGetWorkspaceSize(input.get(), gamma.get(), epsilon,
                                       output.get(), rstd_tensor.get(), &bytes,
                                       &executor),
          label);
  void* workspace = GDN_WORKSPACE(event, bytes, kNormalizationWorkspace);
  CheckNn(aclnnRmsNorm(workspace, bytes, executor, stream), label);
}

void PackedDecodeRecurrentV12(
    statecentric_qwen38_provider_handle_v1* handle, void* query_data,
    void* key_data, void* value_data, void* beta_data, void* gate_data,
    void* recurrent_state_data, void* output_data, aclrtStream stream,
    statecentric_qwen38_event_v1* event) {
  if (handle == nullptr || query_data == nullptr || key_data == nullptr ||
      value_data == nullptr || beta_data == nullptr || gate_data == nullptr ||
      recurrent_state_data == nullptr || output_data == nullptr ||
      stream == nullptr || event == nullptr ||
      handle->packed_decode_actual_lengths == nullptr ||
      handle->packed_decode_state_indices == nullptr) {
    throw std::runtime_error("invalid GDN packed decode recurrent v12 launch");
  }
  constexpr std::int64_t kBatch = 2;
  constexpr std::int64_t kTokens = 2;
  const auto kQueryHeads = static_cast<std::int64_t>(handle->query_heads);
  const auto kValueHeads = static_cast<std::int64_t>(handle->value_heads);
  if (kQueryHeads != 8 || kValueHeads != 24) {
    throw std::runtime_error("GDN packed decode recurrent v12 requires TP2");
  }
  auto query = MakeTensor({kTokens, kQueryHeads, kHeadDim},
                          {kQueryHeads * kHeadDim, kHeadDim, 1},
                          {kTokens, kQueryHeads, kHeadDim}, ACL_BF16,
                          query_data);
  auto key = MakeTensor({kTokens, kQueryHeads, kHeadDim},
                        {kQueryHeads * kHeadDim, kHeadDim, 1},
                        {kTokens, kQueryHeads, kHeadDim}, ACL_BF16, key_data);
  auto value = MakeTensor({kTokens, kValueHeads, kHeadDim},
                          {kValueHeads * kHeadDim, kHeadDim, 1},
                          {kTokens, kValueHeads, kHeadDim}, ACL_BF16,
                          value_data);
  auto beta = MakeTensor({kTokens, kValueHeads}, {kValueHeads, 1},
                         {kTokens, kValueHeads}, ACL_BF16, beta_data);
  auto state = MakeTensor(
      {kBatch, kValueHeads, kHeadDim, kHeadDim},
      {kValueHeads * kHeadDim * kHeadDim, kHeadDim * kHeadDim, kHeadDim, 1},
      {kBatch, kValueHeads, kHeadDim, kHeadDim}, ACL_FLOAT,
      recurrent_state_data);
  auto actual_lengths = MakeTensor(
      {3}, {1}, {3}, ACL_INT32, handle->packed_decode_actual_lengths);
  auto state_indices = MakeTensor(
      {kTokens}, {1}, {kTokens}, ACL_INT32,
      handle->packed_decode_state_indices);
  auto gate = MakeTensor({kTokens, kValueHeads}, {kValueHeads, 1},
                         {kTokens, kValueHeads}, ACL_FLOAT, gate_data);
  auto output = MakeTensor({kTokens, kValueHeads, kHeadDim},
                           {kValueHeads * kHeadDim, kHeadDim, 1},
                           {kTokens, kValueHeads, kHeadDim}, ACL_BF16,
                           output_data);
  std::uint64_t workspace_bytes = 0;
  aclOpExecutor* executor = nullptr;
  CheckNn(aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
              query.get(), key.get(), value.get(), beta.get(), state.get(),
              actual_lengths.get(), state_indices.get(), gate.get(), nullptr,
              nullptr, std::pow(static_cast<float>(kHeadDim), -0.5F),
              output.get(), &workspace_bytes, &executor),
          "aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(GDN packed decode v12)");
  void* workspace = GDN_WORKSPACE(event, workspace_bytes, kRecurrentWorkspace);
  CheckNn(aclnnRecurrentGatedDeltaRule(workspace, workspace_bytes, executor,
                                       stream),
          "aclnnRecurrentGatedDeltaRule(GDN packed decode v12)");
}

void LaneGdnOutputTailV13(
    void* recurrent_output_data, void* z_data, void* gamma_data,
    void* weight_data, void* output_data, std::int64_t tokens,
    std::int64_t value_heads, std::int64_t value, std::int64_t hidden,
    void* deferred_gated_output_data, aclrtStream stream,
    statecentric_qwen38_event_v1* event) {
  if (recurrent_output_data == nullptr || z_data == nullptr ||
      gamma_data == nullptr || weight_data == nullptr ||
      output_data == nullptr || tokens != 1 || value_heads != 24 ||
      value != value_heads * kHeadDim || hidden != kHidden ||
      stream == nullptr || event == nullptr) {
    throw std::runtime_error("invalid GDN lane output tail v13 launch");
  }
  const auto value_rows = tokens * value_heads;
  const auto value_elements =
      static_cast<std::size_t>(tokens * value) * sizeof(std::uint16_t);
  auto recurrent = MakeTensor({value_rows, kHeadDim}, {kHeadDim, 1},
                              {value_rows, kHeadDim}, ACL_BF16,
                              recurrent_output_data);
  auto gamma =
      MakeTensor({kHeadDim}, {1}, {kHeadDim}, ACL_BF16, gamma_data);
  void* normalized_data = Allocate(event, value_elements);
  auto normalized = MakeTensor({value_rows, kHeadDim}, {kHeadDim, 1},
                               {value_rows, kHeadDim}, ACL_BF16,
                               normalized_data);
  RmsNorm(recurrent, gamma, normalized, value_rows, 1.0e-6, stream, event,
          "aclnnRmsNorm(GDN lane output v13)");

  void* activated_data = Allocate(event, value_elements);
  void* gated_data = deferred_gated_output_data != nullptr
                         ? deferred_gated_output_data
                         : Allocate(event, value_elements);
  auto z = MakeTensor({tokens, value}, {value, 1}, {tokens, value}, ACL_BF16,
                      z_data);
  auto activated = MakeTensor({tokens, value}, {value, 1}, {tokens, value},
                              ACL_BF16, activated_data);
  auto normalized_value = MakeTensor({tokens, value}, {value, 1},
                                     {tokens, value}, ACL_BF16,
                                     normalized_data);
  auto gated = MakeTensor({tokens, value}, {value, 1}, {tokens, value},
                          ACL_BF16, gated_data);
  std::uint64_t silu_bytes = 0;
  aclOpExecutor* silu_executor = nullptr;
  CheckNn(aclnnSiluGetWorkspaceSize(z.get(), activated.get(), &silu_bytes,
                                    &silu_executor),
          "aclnnSiluGetWorkspaceSize(GDN lane output v13)");
  void* silu_workspace = GDN_WORKSPACE(event, silu_bytes, kElementwiseWorkspace);
  CheckNn(aclnnSilu(silu_workspace, silu_bytes, silu_executor, stream),
          "aclnnSilu(GDN lane output v13)");
  std::uint64_t mul_bytes = 0;
  aclOpExecutor* mul_executor = nullptr;
  CheckNn(aclnnMulGetWorkspaceSize(activated.get(), normalized_value.get(),
                                   gated.get(), &mul_bytes, &mul_executor),
          "aclnnMulGetWorkspaceSize(GDN lane output v13)");
  void* mul_workspace = GDN_WORKSPACE(event, mul_bytes, kElementwiseWorkspace);
  CheckNn(aclnnMul(mul_workspace, mul_bytes, mul_executor, stream),
          "aclnnMul(GDN lane output v13)");
  if (deferred_gated_output_data != nullptr) return;
  FlatMatmulSharedWeight(gated_data, weight_data, output_data, tokens, value,
                         hidden, stream, event,
                         "aclnnMatmul(GDN lane output projection v13)");
}

void BatchGdnOutputTail(void* recurrent_output_data, void* z_data,
                        void* gamma_data, void* weight_data,
                        void* output_data, std::int64_t batch,
                        std::int64_t tokens, std::int64_t value_heads,
                        std::int64_t value, aclrtStream stream,
                        statecentric_qwen38_event_v1* event) {
  const auto value_rows = tokens * value_heads;
  const auto value_elements = static_cast<std::size_t>(
      batch * tokens * value_heads * kHeadDim);
  auto recurrent = MakeTensor(
      {batch, value_rows, kHeadDim}, {value_rows * kHeadDim, kHeadDim, 1},
      {batch, value_rows, kHeadDim}, ACL_BF16, recurrent_output_data);
  auto gamma = MakeTensor({kHeadDim}, {1}, {kHeadDim}, ACL_BF16, gamma_data);
  void* normalized_data =
      Allocate(event, value_elements * sizeof(std::uint16_t));
  auto normalized = MakeTensor(
      {batch, value_rows, kHeadDim}, {value_rows * kHeadDim, kHeadDim, 1},
      {batch, value_rows, kHeadDim}, ACL_BF16, normalized_data);
  void* rstd_data = Allocate(
      event, static_cast<std::size_t>(batch * value_rows) * sizeof(float));
  auto rstd = MakeTensor({batch, value_rows, 1}, {value_rows, 1, 1},
                         {batch, value_rows, 1}, ACL_FLOAT, rstd_data);
  std::uint64_t norm_bytes = 0;
  aclOpExecutor* norm_executor = nullptr;
  CheckNn(aclnnRmsNormGetWorkspaceSize(
              recurrent.get(), gamma.get(), 1.0e-6, normalized.get(),
              rstd.get(), &norm_bytes, &norm_executor),
          "aclnnRmsNormGetWorkspaceSize(GDN dual-sequence tail v10)");
  void* norm_workspace = GDN_WORKSPACE(event, norm_bytes, kNormalizationWorkspace);
  CheckNn(aclnnRmsNorm(norm_workspace, norm_bytes, norm_executor, stream),
          "aclnnRmsNorm(GDN dual-sequence tail v10)");

  auto z = MakeTensor({batch, tokens, value}, {tokens * value, value, 1},
                      {batch, tokens, value}, ACL_BF16, z_data);
  auto normalized_value = MakeTensor(
      {batch, tokens, value}, {tokens * value, value, 1},
      {batch, tokens, value}, ACL_BF16, normalized_data);
  void* activated_data =
      Allocate(event, value_elements * sizeof(std::uint16_t));
  void* gated_data = Allocate(event, value_elements * sizeof(std::uint16_t));
  auto activated = MakeTensor(
      {batch, tokens, value}, {tokens * value, value, 1},
      {batch, tokens, value}, ACL_BF16, activated_data);
  auto gated = MakeTensor({batch, tokens, value}, {tokens * value, value, 1},
                          {batch, tokens, value}, ACL_BF16, gated_data);
  std::uint64_t silu_bytes = 0;
  aclOpExecutor* silu_executor = nullptr;
  CheckNn(aclnnSiluGetWorkspaceSize(z.get(), activated.get(), &silu_bytes,
                                    &silu_executor),
          "aclnnSiluGetWorkspaceSize(GDN dual-sequence tail v10)");
  void* silu_workspace = GDN_WORKSPACE(event, silu_bytes, kElementwiseWorkspace);
  CheckNn(aclnnSilu(silu_workspace, silu_bytes, silu_executor, stream),
          "aclnnSilu(GDN dual-sequence tail v10)");
  std::uint64_t mul_bytes = 0;
  aclOpExecutor* mul_executor = nullptr;
  CheckNn(aclnnMulGetWorkspaceSize(activated.get(), normalized_value.get(),
                                   gated.get(), &mul_bytes, &mul_executor),
          "aclnnMulGetWorkspaceSize(GDN dual-sequence tail v10)");
  void* mul_workspace = GDN_WORKSPACE(event, mul_bytes, kElementwiseWorkspace);
  CheckNn(aclnnMul(mul_workspace, mul_bytes, mul_executor, stream),
          "aclnnMul(GDN dual-sequence tail v10)");
  FlatMatmulSharedWeight(
      gated_data, weight_data, output_data, batch * tokens, value, kHidden,
      stream, event, "aclnnMatmul(GDN flattened dual-sequence output v10)");
}

void Permute(const std::vector<std::int64_t>& input_shape,
             const std::vector<std::int64_t>& output_shape,
             const std::vector<std::int64_t>& dimensions, aclDataType type,
             void* input, void* output, aclrtStream stream,
             statecentric_qwen38_event_v1* event, const char* label) {
  auto contiguous_strides = [](const std::vector<std::int64_t>& shape) {
    std::vector<std::int64_t> strides(shape.size(), 1);
    for (std::size_t index = shape.size(); index > 1; --index) {
      strides[index - 2] = strides[index - 1] * shape[index - 1];
    }
    return strides;
  };
  auto input_tensor = MakeTensor(input_shape, contiguous_strides(input_shape),
                                 input_shape, type, input);
  auto output_tensor =
      MakeTensor(output_shape, contiguous_strides(output_shape), output_shape,
                 type, output);
  IntArray permutation(
      aclCreateIntArray(dimensions.data(), dimensions.size()));
  if (!permutation) throw std::runtime_error("aclCreateIntArray(GDN permute) failed");
  std::uint64_t bytes = 0;
  aclOpExecutor* executor = nullptr;
  CheckNn(aclnnPermuteGetWorkspaceSize(input_tensor.get(), permutation.get(),
                                        output_tensor.get(), &bytes, &executor),
          label);
  void* workspace = GDN_WORKSPACE(event, bytes, kLayoutWorkspace);
  CheckNn(aclnnPermute(workspace, bytes, executor, stream), label);
}

void SplitQkv(void* qkv, void* query, void* key, void* value,
              std::uint64_t tokens, std::uint32_t qkv_width,
              std::uint32_t query_width, std::uint32_t value_width,
              aclrtStream stream, statecentric_qwen38_event_v1* event) {
  const auto rows = static_cast<std::int64_t>(tokens);
  const auto qkv_columns = static_cast<std::int64_t>(qkv_width);
  const auto query_columns = static_cast<std::int64_t>(query_width);
  const auto value_columns = static_cast<std::int64_t>(value_width);
  auto input = MakeTensor({rows, qkv_columns}, {qkv_columns, 1},
                          {rows, qkv_columns}, ACL_BF16, qkv);
  const auto run_slice = [&](void* destination, std::int64_t columns,
                             std::int64_t start, const char* label) {
    auto output = MakeTensor({rows, columns}, {columns, 1}, {rows, columns},
                             ACL_BF16, destination);
    std::uint64_t workspace_bytes = 0;
    aclOpExecutor* executor = nullptr;
    CheckNn(aclnnSliceGetWorkspaceSize(input.get(), 1, start,
                                       start + columns, 1, output.get(),
                                       &workspace_bytes, &executor),
            label);
    void* workspace = GDN_WORKSPACE(event, workspace_bytes, kLayoutWorkspace);
    CheckNn(aclnnSlice(workspace, workspace_bytes, executor, stream), label);
  };
  run_slice(query, query_columns, 0, "aclnnSlice(GDN Q)");
  run_slice(key, query_columns, query_columns, "aclnnSlice(GDN K)");
  run_slice(value, value_columns, 2 * query_columns, "aclnnSlice(GDN V)");
}

void SplitQkvRepeatable(
    statecentric_qwen38_provider_handle_v1* handle, void* qkv, void* query,
    void* key, void* value, std::uint64_t tokens, std::uint32_t qkv_width,
    std::uint32_t query_width, std::uint32_t value_width, aclrtStream stream,
    statecentric_qwen38_event_v1* event) {
  if (handle == nullptr || event == nullptr || tokens == 0 ||
      tokens > kMaximumSubmissionTokens) {
    throw std::runtime_error("GDN repeatable split geometry is invalid");
  }
  const auto rows = static_cast<std::int64_t>(tokens);
  const auto qkv_columns = static_cast<std::int64_t>(qkv_width);
  const auto query_columns = static_cast<std::int64_t>(query_width);
  const auto value_columns = static_cast<std::int64_t>(value_width);
  statecentric_qwen38_split_cache_v1* selected = nullptr;
  {
    std::scoped_lock lock(handle->split_cache_mutex);
    auto& caches = handle->split_caches.at(tokens);
    const auto available = std::find_if(
        caches.begin(), caches.end(), [](const auto& cache) {
          return cache != nullptr && !cache->in_use;
        });
    if (available != caches.end()) {
      selected = available->get();
      CheckNn(aclSetInputTensorAddr(selected->executor, 0, selected->input,
                                    qkv),
              "aclSetInputTensorAddr(GDN repeatable split)");
      CheckNn(aclSetDynamicOutputTensorAddr(
                  selected->executor, 0, 0, selected->outputs, query),
              "aclSetDynamicOutputTensorAddr(GDN repeatable Q)");
      CheckNn(aclSetDynamicOutputTensorAddr(
                  selected->executor, 0, 1, selected->outputs, key),
              "aclSetDynamicOutputTensorAddr(GDN repeatable K)");
      CheckNn(aclSetDynamicOutputTensorAddr(
                  selected->executor, 0, 2, selected->outputs, value),
              "aclSetDynamicOutputTensorAddr(GDN repeatable V)");
    } else {
      if (caches.size() >= kMaximumConcurrentSplitCachesPerGeometry) {
        throw std::runtime_error(
            "GDN repeatable split concurrency exceeds bounded cache");
      }
      auto cache =
          std::make_unique<statecentric_qwen38_split_cache_v1>();
      cache->input = MakeTensor({rows, qkv_columns}, {qkv_columns, 1},
                                {rows, qkv_columns}, ACL_BF16, qkv)
                         .release();
      cache->query = MakeTensor({rows, query_columns}, {query_columns, 1},
                                {rows, query_columns}, ACL_BF16, query)
                         .release();
      cache->key = MakeTensor({rows, query_columns}, {query_columns, 1},
                              {rows, query_columns}, ACL_BF16, key)
                       .release();
      cache->value = MakeTensor({rows, value_columns}, {value_columns, 1},
                                {rows, value_columns}, ACL_BF16, value)
                         .release();
      const std::array<const aclTensor*, 3> output_tensors{
          cache->query, cache->key, cache->value};
      cache->outputs =
          aclCreateTensorList(output_tensors.data(), output_tensors.size());
      const std::array<std::int64_t, 3> split_sizes{
          query_columns, query_columns, value_columns};
      cache->split_sizes =
          aclCreateIntArray(split_sizes.data(), split_sizes.size());
      if (cache->outputs == nullptr || cache->split_sizes == nullptr) {
        throw std::runtime_error(
            "cannot create GDN repeatable split metadata");
      }
      CheckNn(aclnnSplitWithSizeGetWorkspaceSize(
                  cache->input, cache->split_sizes, 1, cache->outputs,
                  &cache->workspace_bytes, &cache->executor),
              "aclnnSplitWithSizeGetWorkspaceSize(GDN repeatable)");
      CheckNn(aclSetAclOpExecutorRepeatable(cache->executor),
              "aclSetAclOpExecutorRepeatable(GDN split)");
      selected = cache.get();
      caches.push_back(std::move(cache));
    }
    selected->in_use = true;
    event->split_caches.push_back(selected);
  }
  void* workspace = GDN_WORKSPACE(event, selected->workspace_bytes, kLayoutWorkspace);
  CheckNn(aclnnSplitWithSize(workspace, selected->workspace_bytes,
                             selected->executor, stream),
          "aclnnSplitWithSize(GDN repeatable)");
}

void ChunkGdn(statecentric_qwen38_provider_handle_v1* handle,
              std::uint32_t tokens, void* query, void* key, void* value,
              void* beta, void* gate, void* recurrent_state,
              void* recurrent_output, aclrtStream stream,
              statecentric_qwen38_event_v1* event) {
  if (tokens < kChunkKernelMinimumTokens || tokens > kMaximumSubmissionTokens ||
      !handle->cumsum_kernel || !handle->kkt_kernel || !handle->solve_kernel ||
      !handle->merge_kernel || !handle->wy_kernel || handle->ffts == nullptr ||
      handle->ai_cores != 24 || handle->chunk_indices == nullptr) {
    throw std::runtime_error("GDN chunk execution contract is unavailable");
  }
  const auto t = static_cast<std::int64_t>(tokens);
  const auto kQueryHeads = handle->query_heads;
  const auto kValueHeads = handle->value_heads;
  const auto bf16 = sizeof(std::uint16_t);
  const auto key_elements =
      static_cast<std::size_t>(tokens) * kQueryHeads * kHeadDim;
  const auto value_elements =
      static_cast<std::size_t>(tokens) * kValueHeads * kHeadDim;
  const auto gate_elements = static_cast<std::size_t>(tokens) * kValueHeads;
  const auto kkt_elements =
      static_cast<std::size_t>(tokens) * kValueHeads * kChunkSize;
  const auto diagonal_elements =
      static_cast<std::size_t>(tokens) * kValueHeads * 16;
  const auto state_elements =
      static_cast<std::size_t>(kValueHeads * kHeadDim * kHeadDim);

  void* gate_cumsum = Allocate(event, gate_elements * sizeof(float));
  void* beta_hbt = Allocate(event, gate_elements * bf16);
  void* gate_hbt = Allocate(event, gate_elements * sizeof(float));
  void* kkt = Allocate(event, kkt_elements * sizeof(float));
  void* diagonal = Allocate(event, diagonal_elements * sizeof(float));
  void* inverse = Allocate(event, kkt_elements * bf16);
  void* query_bhtd = Allocate(event, key_elements * bf16);
  void* key_bhtd = Allocate(event, key_elements * bf16);
  void* w_bhtd = Allocate(event, value_elements * bf16);
  void* u_bhtd = Allocate(event, value_elements * bf16);
  void* w = handle->wy_outputs_bhtd
                ? w_bhtd
                : Allocate(event, value_elements * bf16);
  void* u = handle->wy_outputs_bhtd
                ? u_bhtd
                : Allocate(event, value_elements * bf16);
  void* chunk_initial_state = Allocate(event, state_elements * sizeof(float));
  void* h = Allocate(event, state_elements * bf16);
  void* v_new = Allocate(event, value_elements * bf16);
  void* chunk_final_state = Allocate(event, state_elements * sizeof(float));
  void* output_bhtd = Allocate(event, value_elements * bf16);
  void* kkt_workspace =
      GDN_WORKSPACE(event, 16'384 * static_cast<std::size_t>(handle->ai_cores), kAotWorkspace);
  void* merge_workspace =
      GDN_WORKSPACE(event, 20'480 * static_cast<std::size_t>(kValueHeads), kAotWorkspace);
  void* wy_workspace = GDN_WORKSPACE(event, 65'536, kAotWorkspace);
  void* cumulative = handle->actual_lengths.at(tokens - 1);

  CumsumArguments cumsum_args{
      handle->ffts, nullptr, nullptr, gate, gate_cumsum, cumulative,
      handle->chunk_indices, static_cast<std::int32_t>(tokens), 1, 1, 1};
  handle->cumsum_kernel->Launch(1, &cumsum_args, sizeof(cumsum_args), stream);
  Permute({1, t, kValueHeads}, {1, kValueHeads, t}, {0, 2, 1}, ACL_BF16,
          beta, beta_hbt, stream, event, "aclnnPermute(GDN beta BTH->HBT)");
  Permute({1, t, kValueHeads}, {1, kValueHeads, t}, {0, 2, 1}, ACL_FLOAT,
          gate_cumsum, gate_hbt, stream, event,
          "aclnnPermute(GDN gate BTH->HBT)");

  KktArguments kkt_args{
      handle->ffts, nullptr, kkt_workspace, key, beta_hbt, gate_hbt, kkt,
      cumulative, handle->chunk_indices, static_cast<std::int32_t>(tokens), 1,
      static_cast<std::int32_t>(handle->ai_cores), 1, 1};
  handle->kkt_kernel->Launch(handle->ai_cores, &kkt_args, sizeof(kkt_args),
                             stream);
  SolveArguments solve_args{
      handle->ffts, nullptr, nullptr, kkt, diagonal, cumulative,
      handle->chunk_indices, static_cast<std::int32_t>(tokens),
      static_cast<std::int32_t>(kValueHeads), 1,
      static_cast<std::int32_t>(kValueHeads), 1};
  handle->solve_kernel->Launch(kValueHeads, &solve_args, sizeof(solve_args),
                               stream);
  MergeArguments merge_args{
      handle->ffts, nullptr, merge_workspace, kkt, diagonal, inverse,
      cumulative, handle->chunk_indices, static_cast<std::int32_t>(tokens),
      static_cast<std::int32_t>(kValueHeads), 1,
      static_cast<std::int32_t>(kValueHeads), 1};
  handle->merge_kernel->Launch(kValueHeads, &merge_args, sizeof(merge_args),
                               stream);
  WyArguments wy_args{
      handle->ffts, nullptr, wy_workspace, key, value, beta_hbt, w, u, inverse,
      gate_hbt, cumulative, handle->chunk_indices,
      static_cast<std::int32_t>(tokens),
      static_cast<std::int32_t>(kValueHeads),
      static_cast<std::int32_t>(kQueryHeads),
      static_cast<std::int32_t>(kHeadDim),
      static_cast<std::int32_t>(kHeadDim), 1, 1, 1};
  handle->wy_kernel->Launch(1, &wy_args, sizeof(wy_args), stream);

  Permute({1, t, kQueryHeads, kHeadDim}, {1, kQueryHeads, t, kHeadDim},
          {0, 2, 1, 3}, ACL_BF16, query, query_bhtd, stream, event,
          "aclnnPermute(GDN query BTHD->BHTD)");
  Permute({1, t, kQueryHeads, kHeadDim}, {1, kQueryHeads, t, kHeadDim},
          {0, 2, 1, 3}, ACL_BF16, key, key_bhtd, stream, event,
          "aclnnPermute(GDN key BTHD->BHTD)");
  if (!handle->wy_outputs_bhtd) {
    Permute({1, t, kValueHeads, kHeadDim}, {1, kValueHeads, t, kHeadDim},
            {0, 2, 1, 3}, ACL_BF16, w, w_bhtd, stream, event,
            "aclnnPermute(GDN W BTHD->BHTD)");
    Permute({1, t, kValueHeads, kHeadDim}, {1, kValueHeads, t, kHeadDim},
            {0, 2, 1, 3}, ACL_BF16, u, u_bhtd, stream, event,
            "aclnnPermute(GDN U BTHD->BHTD)");
  }
  Permute({1, kValueHeads, kHeadDim, kHeadDim},
          {1, kValueHeads, kHeadDim, kHeadDim}, {0, 1, 3, 2}, ACL_FLOAT,
          recurrent_state, chunk_initial_state, stream, event,
          "aclnnPermute(GDN initial state layout)");

  auto query_tensor = MakeTensor(
      {1, kQueryHeads, t, kHeadDim},
      {kQueryHeads * t * kHeadDim, t * kHeadDim, kHeadDim, 1},
      {1, kQueryHeads, t, kHeadDim}, ACL_BF16, query_bhtd);
  auto key_tensor = MakeTensor(
      {1, kQueryHeads, t, kHeadDim},
      {kQueryHeads * t * kHeadDim, t * kHeadDim, kHeadDim, 1},
      {1, kQueryHeads, t, kHeadDim}, ACL_BF16, key_bhtd);
  auto w_tensor = MakeTensor(
      {1, kValueHeads, t, kHeadDim},
      {kValueHeads * t * kHeadDim, t * kHeadDim, kHeadDim, 1},
      {1, kValueHeads, t, kHeadDim}, ACL_BF16, w_bhtd);
  auto u_tensor = MakeTensor(
      {1, kValueHeads, t, kHeadDim},
      {kValueHeads * t * kHeadDim, t * kHeadDim, kHeadDim, 1},
      {1, kValueHeads, t, kHeadDim}, ACL_BF16, u_bhtd);
  auto gate_tensor = MakeTensor({1, kValueHeads, t},
                                {kValueHeads * t, t, 1},
                                {1, kValueHeads, t}, ACL_FLOAT, gate_hbt);
  auto initial_state_tensor = MakeTensor(
      {1, kValueHeads, kHeadDim, kHeadDim},
      {kValueHeads * kHeadDim * kHeadDim, kHeadDim * kHeadDim, kHeadDim, 1},
      {1, kValueHeads, kHeadDim, kHeadDim}, ACL_FLOAT, chunk_initial_state);
  auto h_tensor = MakeTensor(
      {1, kValueHeads, 1, kHeadDim, kHeadDim},
      {kValueHeads * kHeadDim * kHeadDim, kHeadDim * kHeadDim,
       kHeadDim * kHeadDim, kHeadDim, 1},
      {1, kValueHeads, 1, kHeadDim, kHeadDim}, ACL_BF16, h);
  auto v_new_tensor = MakeTensor(
      {1, kValueHeads, t, kHeadDim},
      {kValueHeads * t * kHeadDim, t * kHeadDim, kHeadDim, 1},
      {1, kValueHeads, t, kHeadDim}, ACL_BF16, v_new);
  auto final_state_tensor = MakeTensor(
      {1, kValueHeads, kHeadDim, kHeadDim},
      {kValueHeads * kHeadDim * kHeadDim, kHeadDim * kHeadDim, kHeadDim, 1},
      {1, kValueHeads, kHeadDim, kHeadDim}, ACL_FLOAT, chunk_final_state);
  auto output_tensor = MakeTensor(
      {1, kValueHeads, t, kHeadDim},
      {kValueHeads * t * kHeadDim, t * kHeadDim, kHeadDim, 1},
      {1, kValueHeads, t, kHeadDim}, ACL_BF16, output_bhtd);
  const std::array<std::int64_t, 2> cu_seqlens{0, t};
  const std::array<std::int64_t, 2> chunk_indices{0, 0};
  IntArray cu_array(aclCreateIntArray(cu_seqlens.data(), cu_seqlens.size()));
  IntArray chunk_array(
      aclCreateIntArray(chunk_indices.data(), chunk_indices.size()));
  if (!cu_array || !chunk_array) {
    throw std::runtime_error("aclCreateIntArray(GDN chunk tail) failed");
  }

  std::uint64_t state_workspace_bytes = 0;
  aclOpExecutor* state_executor = nullptr;
  CheckNn(aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
              key_tensor.get(), w_tensor.get(), u_tensor.get(),
              gate_tensor.get(), nullptr, initial_state_tensor.get(), true,
              kChunkSize, true, cu_array.get(), chunk_array.get(), false, false,
              h_tensor.get(), v_new_tensor.get(), final_state_tensor.get(),
              &state_workspace_bytes, &state_executor),
          "aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(GDN)");
  void* state_workspace = GDN_WORKSPACE(event, state_workspace_bytes, kChunkWorkspace);
  CheckNn(aclnnChunkGatedDeltaRuleFwdH(
              state_workspace, state_workspace_bytes, state_executor, stream),
          "aclnnChunkGatedDeltaRuleFwdH(GDN)");

  std::uint64_t output_workspace_bytes = 0;
  aclOpExecutor* output_executor = nullptr;
  CheckNn(aclnnChunkFwdOGetWorkspaceSize(
              query_tensor.get(), key_tensor.get(), v_new_tensor.get(),
              h_tensor.get(), gate_tensor.get(), cu_array.get(),
              chunk_array.get(), std::pow(static_cast<double>(kHeadDim), -0.5),
              kChunkSize, output_tensor.get(), &output_workspace_bytes,
              &output_executor),
          "aclnnChunkFwdOGetWorkspaceSize(GDN)");
  void* output_workspace = GDN_WORKSPACE(event, output_workspace_bytes, kChunkWorkspace);
  CheckNn(aclnnChunkFwdO(output_workspace, output_workspace_bytes,
                         output_executor, stream),
          "aclnnChunkFwdO(GDN)");
  Permute({1, kValueHeads, kHeadDim, kHeadDim},
          {1, kValueHeads, kHeadDim, kHeadDim}, {0, 1, 3, 2}, ACL_FLOAT,
          chunk_final_state, recurrent_state, stream, event,
          "aclnnPermute(GDN final state layout)");
  Permute({1, kValueHeads, t, kHeadDim}, {1, t, kValueHeads, kHeadDim},
          {0, 2, 1, 3}, ACL_BF16, output_bhtd, recurrent_output, stream, event,
          "aclnnPermute(GDN output BHTD->BTHD)");
}

bool TensorGeometry(const statecentric_qwen38_tensor_v1& tensor,
                    std::int32_t device, std::uint32_t scalar,
                    std::initializer_list<std::uint64_t> dimensions,
                    std::initializer_list<std::int64_t> strides,
                    std::size_t element_bytes) {
  if (tensor.size < sizeof(tensor) ||
      tensor.abi_version != STATECENTRIC_QWEN38_PROVIDER_ABI_V1 ||
      tensor.data == 0 || tensor.scalar_type != scalar ||
      tensor.memory_kind != STATECENTRIC_QWEN38_MEMORY_DEVICE ||
      tensor.device_ordinal != device || tensor.rank != dimensions.size()) {
    return false;
  }
  std::uint64_t elements = 1;
  std::size_t index = 0;
  auto stride = strides.begin();
  for (const auto dimension : dimensions) {
    if (tensor.dimensions[index] != dimension ||
        tensor.strides[index] != *stride) return false;
    elements *= dimension;
    ++index;
    ++stride;
  }
  return tensor.byte_length == elements * element_bytes;
}

int32_t Create(const statecentric_qwen38_host_v1* host,
               const statecentric_qwen38_create_v1* create,
               statecentric_qwen38_provider_handle_v1** out_handle,
               statecentric_qwen38_error_v1** out_error) {
  if (host == nullptr || create == nullptr || out_handle == nullptr ||
      host->size < sizeof(*host) ||
      host->abi_version != STATECENTRIC_QWEN38_PROVIDER_ABI_V1 ||
      host->monotonic_time_ns == nullptr || create->size < sizeof(*create) ||
      create->abi_version != STATECENTRIC_QWEN38_PROVIDER_ABI_V1 ||
      create->world_size != 4 || create->rank >= create->world_size ||
      create->device_ordinal != static_cast<std::int32_t>(create->rank) ||
      create->assembly_fingerprint_sha256 == nullptr ||
      create->assembly_fingerprint_length != 32 ||
      create->configuration == nullptr ||
      create->configuration_length <
          offsetof(statecentric_qwen38_gdn_create_config_v1,
                   prefill_qkv_split_variant)) {
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "invalid Qwen3.8 GDN create contract", out_error);
  }
  statecentric_qwen38_gdn_create_config_v1 config{};
  std::memcpy(&config, create->configuration,
              std::min(create->configuration_length, sizeof(config)));
  const bool tp4 = config.qkv_size_per_rank == 2560 &&
                   config.value_size_per_rank == 1536 &&
                   config.gate_heads_per_rank == 12 &&
                   config.query_heads_per_rank == 4 &&
                   config.value_heads_per_rank == 12;
  const bool tp2 = config.qkv_size_per_rank == 5120 &&
                   config.value_size_per_rank == 3072 &&
                   config.gate_heads_per_rank == 24 &&
                   config.query_heads_per_rank == 8 &&
                   config.value_heads_per_rank == 24;
  const auto minimum_config_size =
      offsetof(statecentric_qwen38_gdn_create_config_v1,
               prefill_qkv_split_variant);
  if (config.size < minimum_config_size ||
      config.size > create->configuration_length ||
      config.abi_version != STATECENTRIC_QWEN38_PROVIDER_ABI_V1 ||
      config.hidden_size != kHidden || (!tp4 && !tp2) ||
      config.head_dim != kHeadDim ||
      config.max_tokens_per_submission == 0 ||
      config.max_tokens_per_submission > kMaximumSubmissionTokens ||
      (config.prefill_qkv_split_variant != 0 &&
       config.prefill_qkv_split_variant != 1 &&
       config.prefill_qkv_split_variant != 2 &&
       config.prefill_qkv_split_variant != 3 &&
       config.prefill_qkv_split_variant != 4 &&
       config.prefill_qkv_split_variant != 5 &&
       config.prefill_qkv_split_variant != 6 &&
       config.prefill_qkv_split_variant != 7) ||
      ((config.prefill_qkv_split_variant == 5 ||
        config.prefill_qkv_split_variant == 6 ||
        config.prefill_qkv_split_variant == 7) &&
       (!tp2 || config.max_tokens_per_submission <
                    kChunkKernelMinimumTokens)) ||
      config.reserved != 0) {
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "invalid Qwen3.8 TP4/TP2 GDN configuration", out_error);
  }
  auto* handle = new (std::nothrow) statecentric_qwen38_provider_handle_v1;
  if (handle == nullptr) {
    return Fail(STATECENTRIC_QWEN38_INTERNAL,
                "cannot allocate Qwen3.8 GDN provider", out_error);
  }
  handle->host = host;
  handle->rank = create->rank;
  handle->max_tokens = config.max_tokens_per_submission;
  handle->qkv = config.qkv_size_per_rank;
  handle->value = config.value_size_per_rank;
  handle->gate_heads = config.gate_heads_per_rank;
  handle->query_heads = config.query_heads_per_rank;
  handle->value_heads = config.value_heads_per_rank;
  handle->prefill_qkv_split_variant = config.prefill_qkv_split_variant == 0
                                          ? 1
                                      : (config.prefill_qkv_split_variant == 6 ||
                                         config.prefill_qkv_split_variant == 7)
                                          ? 5
                                          : config.prefill_qkv_split_variant;
  handle->recurrent_prefill_always_recurrent =
      config.prefill_qkv_split_variant == 6;
  handle->wy_outputs_bhtd = config.prefill_qkv_split_variant == 7;
  const char* batched_projection =
      std::getenv("STATECENTRIC_QWEN38_GDN_BATCHED_PROJECTION_V1");
  handle->batched_projection_v1 =
      batched_projection != nullptr &&
      std::strcmp(batched_projection, "1") == 0;
  const char* parallel_batched_projection =
      std::getenv("STATECENTRIC_QWEN38_GDN_PARALLEL_BATCHED_PROJECTION_V2");
  handle->parallel_batched_projection_v2 =
      parallel_batched_projection != nullptr &&
      std::strcmp(parallel_batched_projection, "1") == 0;
  const char* synchronization_event_pool =
      std::getenv("STATECENTRIC_QWEN38_GDN_EVENT_POOL_V3");
  handle->synchronization_event_pool_v3 =
      synchronization_event_pool != nullptr &&
      std::strcmp(synchronization_event_pool, "1") == 0;
  const char* prune_parent_projection_waits =
      std::getenv("STATECENTRIC_QWEN38_GDN_PRUNE_PARENT_WAITS_V4");
  handle->prune_parent_projection_waits_v4 =
      prune_parent_projection_waits != nullptr &&
      std::strcmp(prune_parent_projection_waits, "1") == 0;
  const char* reusable_event_ex =
      std::getenv("STATECENTRIC_QWEN38_GDN_REUSABLE_EVENT_EX_V5");
  handle->reusable_event_ex_v5 =
      reusable_event_ex != nullptr && std::strcmp(reusable_event_ex, "1") == 0;
  const char* event_pool_telemetry =
      std::getenv("STATECENTRIC_QWEN38_GDN_EVENT_POOL_TELEMETRY_V6");
  handle->event_pool_telemetry_v6 =
      event_pool_telemetry != nullptr &&
      std::strcmp(event_pool_telemetry, "1") == 0;
  const char* event_pool_runtime_telemetry =
      std::getenv("STATECENTRIC_QWEN38_GDN_EVENT_POOL_TELEMETRY_V7");
  handle->event_pool_runtime_telemetry_v7 =
      event_pool_runtime_telemetry != nullptr &&
      std::strcmp(event_pool_runtime_telemetry, "1") == 0;
  const char* event_pool_atomic_runtime_telemetry =
      std::getenv("STATECENTRIC_QWEN38_GDN_EVENT_POOL_TELEMETRY_V8");
  handle->event_pool_atomic_runtime_telemetry_v8 =
      event_pool_atomic_runtime_telemetry != nullptr &&
      std::strcmp(event_pool_atomic_runtime_telemetry, "1") == 0;
  const char* event_pool_active_bundles =
      std::getenv("STATECENTRIC_QWEN38_GDN_EVENT_POOL_ACTIVE_BUNDLES_V9");
  if (event_pool_active_bundles != nullptr &&
      std::strlen(event_pool_active_bundles) == 1 &&
      event_pool_active_bundles[0] >= '1' &&
      event_pool_active_bundles[0] <= '4') {
    handle->synchronization_event_pool_active_bundles_v9 =
        static_cast<std::size_t>(event_pool_active_bundles[0] - '0');
  } else if (event_pool_active_bundles != nullptr) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 event-pool active bundle count v9 must be 1-4",
                out_error);
  }
  const char* dual_sequence_tail =
      std::getenv("STATECENTRIC_QWEN38_GDN_DUAL_SEQUENCE_TAIL_V10");
  handle->dual_sequence_tail_v10 =
      dual_sequence_tail != nullptr && std::strcmp(dual_sequence_tail, "1") == 0;
  const char* projection_only_tail =
      std::getenv("STATECENTRIC_QWEN38_GDN_PROJECTION_ONLY_TAIL_V11");
  handle->projection_only_tail_v11 =
      projection_only_tail != nullptr &&
      std::strcmp(projection_only_tail, "1") == 0;
  const char* packed_decode_recurrent =
      std::getenv("STATECENTRIC_QWEN38_GDN_PACKED_DECODE_RECURRENT_V12");
  handle->packed_decode_recurrent_v12 =
      packed_decode_recurrent != nullptr &&
      std::strcmp(packed_decode_recurrent, "1") == 0;
  const char* lane_local_packed_recurrent =
      std::getenv("STATECENTRIC_QWEN38_GDN_LANE_LOCAL_PACKED_RECURRENT_V13");
  handle->lane_local_packed_recurrent_v13 =
      lane_local_packed_recurrent != nullptr &&
      std::strcmp(lane_local_packed_recurrent, "1") == 0;
  const char* lane_anchored_packed_recurrent = std::getenv(
      "STATECENTRIC_QWEN38_GDN_LANE_ANCHORED_PACKED_RECURRENT_V14");
  handle->lane_anchored_packed_recurrent_v14 =
      lane_anchored_packed_recurrent != nullptr &&
      std::strcmp(lane_anchored_packed_recurrent, "1") == 0;
  const char* graph_captured_lane_events = std::getenv(
      "STATECENTRIC_QWEN38_GDN_GRAPH_CAPTURE_EX_SYNC_V15");
  handle->graph_captured_lane_events_v15 =
      graph_captured_lane_events != nullptr &&
      std::strcmp(graph_captured_lane_events, "1") == 0;
  const char* graph_single_follower = std::getenv(
      "STATECENTRIC_QWEN38_GDN_GRAPH_SINGLE_FOLLOWER_V16");
  handle->graph_single_follower_v16 =
      graph_single_follower != nullptr &&
      std::strcmp(graph_single_follower, "1") == 0;
  const char* eager_single_follower = std::getenv(
      "STATECENTRIC_QWEN38_GDN_EAGER_SINGLE_FOLLOWER_V17");
  handle->eager_single_follower_v17 =
      eager_single_follower != nullptr &&
      std::strcmp(eager_single_follower, "1") == 0;
  const char* decode_flat_output_projection = std::getenv(
      "STATECENTRIC_QWEN38_GDN_DECODE_FLAT_OUTPUT_PROJECTION_V18");
  handle->decode_flat_output_projection_v18 =
      decode_flat_output_projection != nullptr &&
      std::strcmp(decode_flat_output_projection, "1") == 0;
  const char* parent_local_decode_tail = std::getenv(
      "STATECENTRIC_QWEN38_GDN_PARENT_LOCAL_DECODE_TAIL_V19");
  handle->parent_local_decode_tail_v19 =
      parent_local_decode_tail != nullptr &&
      std::strcmp(parent_local_decode_tail, "1") == 0;
  if (handle->parallel_batched_projection_v2 &&
      !handle->batched_projection_v1) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 parallel batched projection v2 requires v1",
                out_error);
  }
  if (!statecentric::Qwen38GdnDualSequenceTailFeaturesValid(
          handle->batched_projection_v1, handle->dual_sequence_tail_v10)) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 dual-sequence tail v10 requires batched projection v1",
                out_error);
  }
  if (!statecentric::Qwen38GdnProjectionOnlyTailFeaturesValid(
          handle->batched_projection_v1, handle->dual_sequence_tail_v10,
          handle->projection_only_tail_v11)) {
    delete handle;
    return Fail(
        STATECENTRIC_QWEN38_INVALID_ARGUMENT,
        "Qwen3.8 projection-only tail v11 requires batched projection v1 and "
        "is mutually exclusive with v10",
        out_error);
  }
  if (!statecentric::Qwen38GdnPackedDecodeFeaturesValid(
          handle->batched_projection_v1, handle->dual_sequence_tail_v10,
          handle->projection_only_tail_v11,
          handle->packed_decode_recurrent_v12)) {
    delete handle;
    return Fail(
        STATECENTRIC_QWEN38_INVALID_ARGUMENT,
        "Qwen3.8 packed decode recurrent v12 requires batched projection v1 "
        "and is mutually exclusive with v10/v11",
        out_error);
  }
  if (!statecentric::Qwen38GdnLaneLocalPackedDecodeFeaturesValid(
          handle->batched_projection_v1, handle->dual_sequence_tail_v10,
          handle->projection_only_tail_v11,
          handle->packed_decode_recurrent_v12,
          handle->lane_local_packed_recurrent_v13)) {
    delete handle;
    return Fail(
        STATECENTRIC_QWEN38_INVALID_ARGUMENT,
        "Qwen3.8 lane-local packed recurrent v13 requires batched projection "
        "v1 and is mutually exclusive with v10/v11/v12",
        out_error);
  }
  if (!statecentric::Qwen38GdnLaneAnchoredPackedDecodeFeaturesValid(
          handle->batched_projection_v1, handle->dual_sequence_tail_v10,
          handle->projection_only_tail_v11,
          handle->packed_decode_recurrent_v12,
          handle->lane_local_packed_recurrent_v13,
          handle->lane_anchored_packed_recurrent_v14)) {
    delete handle;
    return Fail(
        STATECENTRIC_QWEN38_INVALID_ARGUMENT,
        "Qwen3.8 lane-anchored packed recurrent v14 requires batched "
        "projection v1 and is mutually exclusive with v10/v11/v12/v13",
        out_error);
  }
  if (!statecentric::Qwen38GdnGraphCapturedLaneEventsFeaturesValid(
          handle->lane_anchored_packed_recurrent_v14,
          handle->graph_captured_lane_events_v15)) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 graph-captured lane events v15 require "
                "lane-anchored packed recurrent v14",
                out_error);
  }
  if (!statecentric::Qwen38GdnGraphSingleFollowerFeaturesValid(
          handle->graph_captured_lane_events_v15,
          handle->graph_single_follower_v16)) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 graph single-follower v16 requires "
                "graph-captured lane events v15",
                out_error);
  }
  if (!statecentric::Qwen38GdnEagerSingleFollowerFeaturesValid(
          handle->graph_single_follower_v16,
          handle->eager_single_follower_v17)) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 eager single-follower v17 requires graph "
                "single-follower v16",
                out_error);
  }
  if (!statecentric::Qwen38GdnDecodeFlatOutputProjectionFeaturesValid(
          handle->eager_single_follower_v17,
          handle->decode_flat_output_projection_v18)) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 decode flat output projection v18 requires eager "
                "single-follower v17",
                out_error);
  }
  if (!statecentric::Qwen38GdnParentLocalDecodeTailFeaturesValid(
          handle->decode_flat_output_projection_v18,
          handle->parent_local_decode_tail_v19)) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 parent-local decode tail v19 requires decode flat "
                "output projection v18",
                out_error);
  }
  if ((handle->synchronization_event_pool_v3 ||
       handle->prune_parent_projection_waits_v4) &&
      !handle->parallel_batched_projection_v2) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 event policy extensions require parallel v2",
                out_error);
  }
  if (handle->reusable_event_ex_v5 &&
      !handle->synchronization_event_pool_v3) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 reusable Ex event v5 requires event pool v3",
                out_error);
  }
  if (handle->event_pool_telemetry_v6 &&
      !handle->synchronization_event_pool_v3) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 event-pool telemetry v6 requires event pool v3",
                out_error);
  }
  if (handle->event_pool_runtime_telemetry_v7 &&
      !handle->synchronization_event_pool_v3) {
    delete handle;
    return Fail(
        STATECENTRIC_QWEN38_INVALID_ARGUMENT,
        "Qwen3.8 event-pool runtime telemetry v7 requires event pool v3",
        out_error);
  }
  if (handle->event_pool_atomic_runtime_telemetry_v8 &&
      !handle->synchronization_event_pool_v3) {
    delete handle;
    return Fail(
        STATECENTRIC_QWEN38_INVALID_ARGUMENT,
        "Qwen3.8 event-pool atomic runtime telemetry v8 requires event pool v3",
        out_error);
  }
  if (handle->synchronization_event_pool_active_bundles_v9 !=
          statecentric::kQwen38SynchronizationEventBundleCount &&
      !handle->synchronization_event_pool_v3) {
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 event-pool active bundle count v9 requires event pool v3",
                out_error);
  }
  try {
    if (handle->max_tokens >= kChunkKernelMinimumTokens) {
      InitializeChunkKernels(handle, create->device_ordinal);
    }
    if (handle->prefill_qkv_split_variant == 5) {
      InitializePostConvPackKernel(handle, create->device_ordinal);
    }
#ifdef STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1
    handle->prefill_tensor_pool = std::make_unique<statecentric::CompletedBufferPool>(
        64U * 1024U * 1024U, 8U * 1024U * 1024U,
        [](std::size_t bytes) {
          void* pointer = nullptr;
          CheckAcl(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
                   "aclrtMalloc(GDN prefill tensor pool v1)");
          return pointer;
        }, [](void* pointer) { return aclrtFree(pointer) == ACL_SUCCESS; });
#endif
  } catch (const std::exception& error) {
    FreeHandleConstants(handle);
    delete handle;
    return Fail(STATECENTRIC_QWEN38_INTERNAL,
                std::string("Qwen3.8 GDN AOT startup failed: ") + error.what(),
                out_error);
  }
  *out_handle = handle;
  return STATECENTRIC_QWEN38_OK;
}

void Destroy(statecentric_qwen38_provider_handle_v1* handle) {
  FreeHandleConstants(handle);
#ifdef STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1
  if (handle != nullptr && handle->prefill_tensor_pool != nullptr) {
    const bool drained = handle->prefill_tensor_pool->Close();
    const auto stats = handle->prefill_tensor_pool->Snapshot();
    std::fprintf(stderr,
        "STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1={\"rank\":%u,"
        "\"drained\":%s,\"owned_bytes\":%zu,\"peak_owned_bytes\":%zu,"
        "\"active_leases\":%zu,\"quarantined_buffers\":%zu,"
        "\"allocations\":%llu,\"reuses\":%llu,\"budget_bypasses\":%llu}\n",
        handle->rank, drained ? "true" : "false", stats.owned_bytes,
        stats.peak_owned_bytes, stats.active_leases, stats.quarantined_buffers,
        static_cast<unsigned long long>(stats.allocations),
        static_cast<unsigned long long>(stats.reuses),
        static_cast<unsigned long long>(stats.budget_bypasses));
  }
#endif
  delete handle;
}

int32_t Submit(statecentric_qwen38_provider_handle_v1* handle,
               const statecentric_qwen38_execution_v1* execution,
               std::uint64_t deadline_monotonic_ns,
               statecentric_qwen38_event_v1** out_event,
               statecentric_qwen38_error_v1** out_error) {
  if (handle == nullptr || execution == nullptr || out_event == nullptr ||
      execution->size < sizeof(*execution) ||
      execution->abi_version != STATECENTRIC_QWEN38_PROVIDER_ABI_V1 ||
      execution->rank != handle->rank || execution->world_size != 4 ||
      execution->device_ordinal != static_cast<std::int32_t>(handle->rank) ||
      execution->stage != STATECENTRIC_QWEN38_STAGE_GDN ||
      (execution->flags != STATECENTRIC_QWEN38_GDN_FORWARD_V1 &&
       execution->flags != (STATECENTRIC_QWEN38_GDN_FORWARD_V1 |
                            STATECENTRIC_QWEN38_GDN_PREFILL_SPLIT_V2) &&
       execution->flags != (STATECENTRIC_QWEN38_GDN_FORWARD_V1 |
                            STATECENTRIC_QWEN38_GDN_PREFILL_SPLIT_V3) &&
       execution->flags !=
           (STATECENTRIC_QWEN38_GDN_FORWARD_V1 |
            STATECENTRIC_QWEN38_GDN_PREFILL_STRIDED_QKV_V4) &&
       execution->flags !=
           (STATECENTRIC_QWEN38_GDN_FORWARD_V1 |
            STATECENTRIC_QWEN38_GDN_PREFILL_POST_CONV_PACK_V5)) ||
      execution->layer_index >= 64 || execution->layer_index % 4 == 3 ||
      execution->token_count == 0 ||
      execution->token_count > handle->max_tokens || execution->stream == 0 ||
      execution->input_count != 10 || execution->output_count != 1 ||
      execution->mutable_state_count != 2 || execution->inputs == nullptr ||
      execution->outputs == nullptr || execution->mutable_state == nullptr) {
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "invalid Qwen3.8 GDN submission", out_error);
  }
  const auto tokens = static_cast<std::uint64_t>(execution->token_count);
  const bool prefill_split_v2 =
      (execution->flags & STATECENTRIC_QWEN38_GDN_PREFILL_SPLIT_V2) != 0;
  const bool prefill_split_v3 =
      (execution->flags & STATECENTRIC_QWEN38_GDN_PREFILL_SPLIT_V3) != 0;
  const bool prefill_strided_qkv_v4 =
      (execution->flags & STATECENTRIC_QWEN38_GDN_PREFILL_STRIDED_QKV_V4) != 0;
  const bool prefill_post_conv_pack_v5 =
      (execution->flags & STATECENTRIC_QWEN38_GDN_PREFILL_POST_CONV_PACK_V5) !=
      0;
  const bool decode_forward_v1 =
      execution->flags == STATECENTRIC_QWEN38_GDN_FORWARD_V1;
  const bool packed_decode_layout_ready =
      statecentric::Qwen38GdnPackedDecodeLayoutReady(
          prefill_post_conv_pack_v5, decode_forward_v1);
  if ((prefill_split_v2 && handle->prefill_qkv_split_variant != 2) ||
      (prefill_split_v3 && handle->prefill_qkv_split_variant != 3) ||
      (prefill_strided_qkv_v4 &&
       handle->prefill_qkv_split_variant != 4) ||
      (prefill_post_conv_pack_v5 &&
       handle->prefill_qkv_split_variant != 5)) {
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "Qwen3.8 GDN prefill split variant was not admitted",
                out_error);
  }
  const auto kQkv = handle->qkv;
  const auto kValue = handle->value;
  const auto kGate = handle->gate_heads;
  const auto kQueryHeads = handle->query_heads;
  const auto kValueHeads = handle->value_heads;
  const auto device = execution->device_ordinal;
  const auto& input = execution->inputs;
  const auto& output = execution->outputs[0];
  const auto& conv_state = execution->mutable_state[0];
  const auto& recurrent_state = execution->mutable_state[1];
  const auto state_count = conv_state.tensor.dimensions[0];
  const bool multistate = state_count > 1;
  const statecentric_qwen38_gdn_batch_v1* batch = nullptr;
  const statecentric_qwen38_precomputed_projection_v1* precomputed = nullptr;
  std::span<const std::uint32_t> token_offsets;
  std::span<const std::uint32_t> token_counts;
  if (multistate) {
    if (execution->workspace == 0 ||
        execution->workspace_length != sizeof(statecentric_qwen38_gdn_batch_v1)) {
      return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                  "Qwen3.8 multi-state GDN metadata is missing", out_error);
    }
    batch = reinterpret_cast<const statecentric_qwen38_gdn_batch_v1*>(
        execution->workspace);
    if (batch->size < sizeof(*batch) ||
        batch->abi_version != STATECENTRIC_QWEN38_PROVIDER_ABI_V1 ||
        batch->state_count != state_count || batch->reserved != 0 ||
        batch->token_offsets == nullptr || batch->token_counts == nullptr) {
      return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                  "Qwen3.8 multi-state GDN metadata is invalid", out_error);
    }
    token_offsets = {batch->token_offsets, state_count};
    token_counts = {batch->token_counts, state_count};
    std::uint64_t expected_offset = 0;
    for (std::size_t index = 0; index < state_count; ++index) {
      if (token_offsets[index] != expected_offset || token_counts[index] == 0) {
        return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                    "Qwen3.8 multi-state GDN ranges are not packed", out_error);
      }
      expected_offset += token_counts[index];
    }
    if (expected_offset != tokens) {
      return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                  "Qwen3.8 multi-state GDN ranges do not cover tokens",
                  out_error);
    }
  } else if (execution->workspace != 0 || execution->workspace_length != 0) {
    if (execution->workspace == 0 ||
        execution->workspace_length != sizeof(*precomputed)) {
      return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                  "Qwen3.8 single-state GDN metadata is invalid", out_error);
    }
    precomputed =
        reinterpret_cast<const statecentric_qwen38_precomputed_projection_v1*>(
            execution->workspace);
    if (precomputed->magic != kPrecomputedProjectionMagic ||
        precomputed->tokens != tokens || precomputed->reserved != 0 ||
        precomputed->defer_output_tail > 1 ||
        precomputed->defer_output_projection > 1 ||
        precomputed->use_precomputed_projection > 1 ||
        precomputed->defer_recurrent > 1 ||
        (precomputed->defer_output_tail == 1 &&
         precomputed->defer_output_projection == 1) ||
        (precomputed->defer_recurrent == 1 &&
         (precomputed->defer_output_tail == 1 ||
          precomputed->defer_output_projection == 1)) ||
        ((precomputed->qkv == nullptr) != (precomputed->z == nullptr)) ||
        (precomputed->use_precomputed_projection == 1 &&
         (precomputed->qkv == nullptr || precomputed->z == nullptr)) ||
        (precomputed->defer_output_tail == 1 &&
         precomputed->recurrent_output == nullptr) ||
        (precomputed->defer_output_projection == 1 &&
         precomputed->gated_output == nullptr) ||
        (precomputed->defer_recurrent == 1 &&
         (precomputed->query_norm == nullptr ||
          precomputed->key_norm == nullptr || precomputed->value == nullptr ||
          precomputed->beta == nullptr || precomputed->gate == nullptr))) {
      return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                  "Qwen3.8 precomputed projection metadata is invalid",
                  out_error);
    }
  }
  const bool tensors_valid =
      state_count > 0 && state_count <= 4 &&
      (!multistate || state_count <= tokens) &&
      TensorGeometry(input[0], device, STATECENTRIC_QWEN38_SCALAR_BF16,
                     {tokens, kHidden}, {kHidden, 1}, 2) &&
      TensorGeometry(input[1], device, STATECENTRIC_QWEN38_SCALAR_BF16,
                     {kHidden, kQkv}, {1, kHidden}, 2) &&
      TensorGeometry(input[2], device, STATECENTRIC_QWEN38_SCALAR_BF16,
                     {kHidden, kValue}, {1, kHidden}, 2) &&
      TensorGeometry(input[3], device, STATECENTRIC_QWEN38_SCALAR_BF16,
                     {kHidden, kGate}, {1, kHidden}, 2) &&
      TensorGeometry(input[4], device, STATECENTRIC_QWEN38_SCALAR_BF16,
                     {kHidden, kGate}, {1, kHidden}, 2) &&
      TensorGeometry(input[5], device, STATECENTRIC_QWEN38_SCALAR_BF16,
                     {4, kQkv}, {kQkv, 1}, 2) &&
      TensorGeometry(input[6], device, STATECENTRIC_QWEN38_SCALAR_F32,
                     {kGate}, {1}, 4) &&
      TensorGeometry(input[7], device, STATECENTRIC_QWEN38_SCALAR_F32,
                     {kGate}, {1}, 4) &&
      TensorGeometry(input[8], device, STATECENTRIC_QWEN38_SCALAR_BF16,
                     {kHeadDim}, {1}, 2) &&
      TensorGeometry(input[9], device, STATECENTRIC_QWEN38_SCALAR_BF16,
                     {kValue, kHidden}, {1, kValue}, 2) &&
      TensorGeometry(output, device, STATECENTRIC_QWEN38_SCALAR_BF16,
                     {tokens, kHidden}, {kHidden, 1}, 2) &&
      conv_state.size >= sizeof(conv_state) &&
      conv_state.abi_version == STATECENTRIC_QWEN38_PROVIDER_ABI_V1 &&
      conv_state.layer_index == execution->layer_index &&
      conv_state.state_kind != nullptr &&
      std::strcmp(conv_state.state_kind, "gdn.conv") == 0 &&
      TensorGeometry(conv_state.tensor, device, STATECENTRIC_QWEN38_SCALAR_BF16,
                     {state_count, kConvHistory, kQkv},
                     {kConvHistory * kQkv, kQkv, 1}, 2) &&
      recurrent_state.size >= sizeof(recurrent_state) &&
      recurrent_state.abi_version == STATECENTRIC_QWEN38_PROVIDER_ABI_V1 &&
      recurrent_state.layer_index == execution->layer_index &&
      recurrent_state.state_kind != nullptr &&
      std::strcmp(recurrent_state.state_kind, "gdn.recurrent") == 0 &&
      TensorGeometry(recurrent_state.tensor, device,
                     STATECENTRIC_QWEN38_SCALAR_F32,
                     {state_count, kValueHeads, kHeadDim, kHeadDim},
                     {kValueHeads * kHeadDim * kHeadDim,
                      kHeadDim * kHeadDim, kHeadDim, 1}, 4);
  if (!tensors_valid) {
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "invalid Qwen3.8 GDN tensor or state geometry", out_error);
  }
  const auto started = handle->host->monotonic_time_ns(handle->host->context);
  if (deadline_monotonic_ns <= started) {
    return Fail(STATECENTRIC_QWEN38_TIMEOUT,
                "Qwen3.8 GDN deadline expired", out_error);
  }
  const auto packed_decode_plan = statecentric::PlanQwen38GdnPackedDecode(
      handle->packed_decode_recurrent_v12, multistate,
      packed_decode_layout_ready, handle->batched_projection_v1, state_count,
      tokens, token_counts);
  const auto lane_local_packed_decode_plan =
      statecentric::PlanQwen38GdnLaneLocalPackedDecode(
          handle->lane_local_packed_recurrent_v13, multistate,
          packed_decode_layout_ready, handle->batched_projection_v1,
          state_count, tokens, token_counts);
  const auto lane_anchored_packed_decode_plan =
      statecentric::PlanQwen38GdnLaneAnchoredPackedDecode(
          handle->lane_anchored_packed_recurrent_v14, multistate,
          packed_decode_layout_ready, handle->batched_projection_v1,
          state_count, tokens, token_counts);
  if (handle->packed_decode_recurrent_v12) {
    handle->packed_decode_submit_calls_v12.fetch_add(1,
                                                      std::memory_order_relaxed);
    if (packed_decode_plan.packed_recurrent) {
      handle->packed_decode_admitted_calls_v12.fetch_add(
          1, std::memory_order_relaxed);
      bool expected = false;
      if (handle->packed_decode_admission_emitted_v12.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel,
              std::memory_order_relaxed)) {
        EmitPackedDecodeFirstAdmissionV12(handle);
      }
    }
  }
  if (handle->lane_anchored_packed_recurrent_v14) {
    handle->lane_anchored_submit_calls_v14.fetch_add(
        1, std::memory_order_relaxed);
    if (lane_anchored_packed_decode_plan.lane_prepare &&
        lane_anchored_packed_decode_plan.packed_recurrent &&
        lane_anchored_packed_decode_plan.lane_output_tail) {
      handle->lane_anchored_admitted_calls_v14.fetch_add(
          1, std::memory_order_relaxed);
      bool expected = false;
      if (handle->lane_anchored_admission_emitted_v14.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel,
              std::memory_order_relaxed)) {
        EmitLaneAnchoredFirstAdmissionV14(handle);
      }
    }
  }
  const bool lane_packed_decode_admitted =
      lane_local_packed_decode_plan.packed_recurrent ||
      lane_anchored_packed_decode_plan.packed_recurrent;
  if (multistate &&
      (prefill_post_conv_pack_v5 || lane_packed_decode_admitted) &&
      !packed_decode_plan.packed_recurrent) {
    try {
      EnsureSequenceStreams(handle);
    } catch (const std::exception& error) {
      return Fail(
          STATECENTRIC_QWEN38_INTERNAL,
          std::string("Qwen3.8 sequence stream initialization failed: ") +
              error.what(),
          out_error);
    }
    const auto stream_plan =
        statecentric::PlanQwen38GdnSequenceStreams(state_count);
    if (!stream_plan.has_value()) {
      return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                  "Qwen3.8 sequence stream plan is unavailable", out_error);
    }
    auto* aggregate = new (std::nothrow) statecentric_qwen38_event_v1;
    if (aggregate == nullptr) {
      return Fail(STATECENTRIC_QWEN38_INTERNAL,
                  "cannot allocate Qwen3.8 sequence-normalized GDN event",
                  out_error);
    }
    aggregate->stream = reinterpret_cast<aclrtStream>(execution->stream);
    aggregate->submitted_ns = started;
    if (handle->graph_captured_lane_events_v15 &&
        lane_anchored_packed_decode_plan.packed_recurrent) {
      try {
        aggregate->model_ri_cross_stream_capture =
            GraphCaptureActive(aggregate->stream);
      } catch (const std::exception& error) {
        delete aggregate;
        return Fail(STATECENTRIC_QWEN38_INTERNAL,
                    std::string("Qwen3.8 model-RI capture query failed: ") +
                        error.what(),
                    out_error);
      }
      if (aggregate->model_ri_cross_stream_capture) {
        handle->graph_captured_lane_admissions_v15.fetch_add(
            1, std::memory_order_relaxed);
        bool expected = false;
        if (handle->graph_captured_lane_admission_emitted_v15
                .compare_exchange_strong(expected, true,
                                         std::memory_order_acq_rel,
                                         std::memory_order_relaxed)) {
          EmitGraphCapturedLaneFirstAdmissionV15(handle);
        }
      }
    }
    aggregate->sequence_events.reserve(state_count);
    const auto hidden_row_bytes = static_cast<std::size_t>(kHidden) *
                                  sizeof(std::uint16_t);
    const auto conv_state_bytes =
        static_cast<std::size_t>(kConvHistory * kQkv) * sizeof(std::uint16_t);
    const auto recurrent_state_bytes =
        static_cast<std::size_t>(kValueHeads * kHeadDim * kHeadDim) *
        sizeof(float);
    const auto projection_plan = statecentric::PlanQwen38GdnBatchedProjection(
        handle->batched_projection_v1,
        handle->parallel_batched_projection_v2, state_count, tokens,
        token_counts);
    const auto tail_plan = statecentric::PlanQwen38GdnDualSequenceTail(
        handle->dual_sequence_tail_v10, multistate,
        prefill_post_conv_pack_v5, handle->batched_projection_v1, state_count,
        tokens, token_counts);
    const auto projection_tail_plan =
        statecentric::PlanQwen38GdnProjectionOnlyTail(
            handle->projection_only_tail_v11, multistate,
            prefill_post_conv_pack_v5, handle->batched_projection_v1,
            state_count, tokens, token_counts);
    const bool batch_project_qkv_z = projection_plan.qkv && projection_plan.z;
    const bool batch_output_tail = tail_plan.batch_output_projection;
    const bool batch_output_projection_only =
        projection_tail_plan.batch_output_projection;
    const bool lane_local_packed_recurrent =
        lane_local_packed_decode_plan.lane_prepare &&
        lane_local_packed_decode_plan.packed_recurrent &&
        lane_local_packed_decode_plan.lane_output_tail;
    const bool lane_anchored_packed_recurrent =
        lane_anchored_packed_decode_plan.lane_prepare &&
        lane_anchored_packed_decode_plan.packed_recurrent &&
        lane_anchored_packed_decode_plan.lane_output_tail;
    const bool lane_packed_recurrent =
        lane_local_packed_recurrent || lane_anchored_packed_recurrent;
    const bool decode_flat_output_projection =
        handle->decode_flat_output_projection_v18 &&
        lane_anchored_packed_recurrent &&
        !aggregate->model_ri_cross_stream_capture;
    const bool parent_local_decode_tail =
        handle->parent_local_decode_tail_v19 &&
        decode_flat_output_projection;
    const auto single_follower_stream_plan =
        statecentric::PlanQwen38GdnSingleFollowerStreams(
            aggregate->model_ri_cross_stream_capture,
            handle->graph_single_follower_v16,
            handle->eager_single_follower_v17,
            static_cast<unsigned>(state_count));
    if (single_follower_stream_plan.anchor_on_parent_stream &&
        aggregate->model_ri_cross_stream_capture) {
      handle->graph_single_follower_admissions_v16.fetch_add(
          1, std::memory_order_relaxed);
      bool expected = false;
      if (handle->graph_single_follower_admission_emitted_v16
              .compare_exchange_strong(expected, true,
                                       std::memory_order_acq_rel,
                                       std::memory_order_relaxed)) {
        EmitGraphSingleFollowerFirstAdmissionV16(handle);
      }
    }
    if (single_follower_stream_plan.anchor_on_parent_stream &&
        !aggregate->model_ri_cross_stream_capture &&
        handle->eager_single_follower_v17) {
      handle->eager_single_follower_admissions_v17.fetch_add(
          1, std::memory_order_relaxed);
      bool expected = false;
      if (handle->eager_single_follower_admission_emitted_v17
              .compare_exchange_strong(expected, true,
                                       std::memory_order_acq_rel,
                                       std::memory_order_relaxed)) {
        EmitEagerSingleFollowerFirstAdmissionV17(handle);
      }
    }
    if (decode_flat_output_projection) {
      handle->decode_flat_output_admissions_v18.fetch_add(
          1, std::memory_order_relaxed);
      bool expected = false;
      if (handle->decode_flat_output_admission_emitted_v18
              .compare_exchange_strong(expected, true,
                                       std::memory_order_acq_rel,
                                       std::memory_order_relaxed)) {
        EmitDecodeFlatOutputFirstAdmissionV18(handle);
      }
    }
    if (parent_local_decode_tail) {
      handle->parent_local_decode_tail_admissions_v19.fetch_add(
          1, std::memory_order_relaxed);
      bool expected = false;
      if (handle->parent_local_decode_tail_admission_emitted_v19
              .compare_exchange_strong(expected, true,
                                       std::memory_order_acq_rel,
                                       std::memory_order_relaxed)) {
        EmitParentLocalDecodeTailFirstAdmissionV19(handle);
      }
    }
    const auto anchor_stream =
        single_follower_stream_plan.anchor_on_parent_stream
            ? aggregate->stream
            : handle->sequence_streams.at(stream_plan->lane_by_sequence[0]);
    const auto follower_stream = handle->sequence_streams.at(
        stream_plan->lane_by_sequence[1]);
    const bool parallel_project_qkv_z =
        batch_project_qkv_z && projection_plan.parallel_qkv_z;
    void* batched_qkv = nullptr;
    void* batched_z = nullptr;
    void* batched_recurrent_output = nullptr;
    void* batched_gated_output = nullptr;
    void* batched_query_norm = nullptr;
    void* batched_key_norm = nullptr;
    void* batched_value = nullptr;
    void* batched_beta = nullptr;
    void* batched_gate = nullptr;
    try {
      const auto graph_event_plan = statecentric::PlanQwen38GdnGraphEvents(
          aggregate->model_ri_cross_stream_capture,
          handle->synchronization_event_pool_v3);
      if (graph_event_plan.acquire_reusable_pool) {
        aggregate->synchronization_event_bundle =
            AcquireSynchronizationEventBundle(handle);
      }
      if (batch_project_qkv_z || lane_packed_recurrent) {
        batched_qkv = Allocate(
            aggregate, static_cast<std::size_t>(tokens * kQkv) *
                           sizeof(std::uint16_t));
        batched_z = Allocate(
            aggregate, static_cast<std::size_t>(tokens * kValue) *
                           sizeof(std::uint16_t));
      }
      if (lane_packed_recurrent) {
        batched_query_norm = Allocate(
            aggregate,
            static_cast<std::size_t>(tokens * kQueryHeads * kHeadDim) *
                sizeof(std::uint16_t));
        batched_key_norm = Allocate(
            aggregate,
            static_cast<std::size_t>(tokens * kQueryHeads * kHeadDim) *
                sizeof(std::uint16_t));
        batched_value = Allocate(
            aggregate, static_cast<std::size_t>(tokens * kValue) *
                           sizeof(std::uint16_t));
        batched_beta = Allocate(
            aggregate, static_cast<std::size_t>(tokens * kGate) *
                           sizeof(std::uint16_t));
        batched_gate = Allocate(
            aggregate,
            static_cast<std::size_t>(tokens * kGate) * sizeof(float));
        batched_recurrent_output = Allocate(
            aggregate, static_cast<std::size_t>(tokens * kValue) *
                           sizeof(std::uint16_t));
      }
      if (batch_output_tail) {
        batched_recurrent_output = Allocate(
            aggregate, static_cast<std::size_t>(tokens * kValue) *
                           sizeof(std::uint16_t));
      }
      if (batch_output_projection_only ||
          (decode_flat_output_projection && !parent_local_decode_tail)) {
        batched_gated_output = Allocate(
            aggregate, static_cast<std::size_t>(tokens * kValue) *
                           sizeof(std::uint16_t));
      }
      const auto input_ready = SynchronizationEvent(
          aggregate, 0, "aclrtCreateEvent(GDN sequence input ready)");
      aclrtEvent qkv_ready = nullptr;
      aclrtEvent z_ready = nullptr;
      if (parallel_project_qkv_z) {
        qkv_ready = SynchronizationEvent(
            aggregate, 1, "aclrtCreateEvent(GDN batched QKV ready)");
        z_ready = SynchronizationEvent(
            aggregate, 2, "aclrtCreateEvent(GDN batched Z ready)");
        const auto qkv_stream = handle->sequence_streams.at(2);
        const auto z_stream = handle->sequence_streams.at(3);
        if (qkv_stream == nullptr || z_stream == nullptr) {
          throw std::runtime_error(
              "Qwen3.8 parallel projection lane stream is missing");
        }
        CheckAcl(aclrtRecordEvent(input_ready, aggregate->stream),
                 "aclrtRecordEvent(GDN projection input ready)");
        CheckAcl(aclrtStreamWaitEvent(qkv_stream, input_ready),
                 "aclrtStreamWaitEvent(GDN QKV projection input ready)");
        CheckAcl(aclrtStreamWaitEvent(z_stream, input_ready),
                 "aclrtStreamWaitEvent(GDN Z projection input ready)");
        BatchMatmulSharedWeight(
            reinterpret_cast<void*>(input[0].data),
            reinterpret_cast<void*>(input[1].data), batched_qkv,
            projection_plan.batch, projection_plan.tokens_per_sequence,
            kHidden, kQkv, qkv_stream, aggregate,
            "aclnnBatchMatMul(GDN parallel batched QKV)");
        CheckAcl(aclrtRecordEvent(qkv_ready, qkv_stream),
                 "aclrtRecordEvent(GDN batched QKV ready)");
        BatchMatmulSharedWeight(
            reinterpret_cast<void*>(input[0].data),
            reinterpret_cast<void*>(input[2].data), batched_z,
            projection_plan.batch, projection_plan.tokens_per_sequence,
            kHidden, kValue, z_stream, aggregate,
            "aclnnBatchMatMul(GDN parallel batched Z)");
        CheckAcl(aclrtRecordEvent(z_ready, z_stream),
                 "aclrtRecordEvent(GDN batched Z ready)");
        if (!handle->prune_parent_projection_waits_v4) {
          CheckAcl(aclrtStreamWaitEvent(aggregate->stream, qkv_ready),
                   "aclrtStreamWaitEvent(GDN parent QKV ready)");
          CheckAcl(aclrtStreamWaitEvent(aggregate->stream, z_ready),
                   "aclrtStreamWaitEvent(GDN parent Z ready)");
        }
      } else {
        if (batch_project_qkv_z) {
          BatchMatmulSharedWeight(
              reinterpret_cast<void*>(input[0].data),
              reinterpret_cast<void*>(input[1].data), batched_qkv,
              projection_plan.batch, projection_plan.tokens_per_sequence,
              kHidden, kQkv, aggregate->stream, aggregate,
              "aclnnBatchMatMul(GDN batched QKV)");
          BatchMatmulSharedWeight(
              reinterpret_cast<void*>(input[0].data),
              reinterpret_cast<void*>(input[2].data), batched_z,
              projection_plan.batch, projection_plan.tokens_per_sequence,
              kHidden, kValue, aggregate->stream, aggregate,
              "aclnnBatchMatMul(GDN batched Z)");
        }
        CheckAcl(aclrtRecordEvent(input_ready, aggregate->stream),
                 "aclrtRecordEvent(GDN sequence input ready)");
      }
      for (std::size_t sequence = 0; sequence < state_count; ++sequence) {
        const auto lane = stream_plan->lane_by_sequence[sequence];
        const auto sequence_stream =
            single_follower_stream_plan.anchor_on_parent_stream &&
                    sequence == 0
                ? aggregate->stream
                : handle->sequence_streams.at(lane);
        if (sequence_stream == nullptr) {
          throw std::runtime_error("Qwen3.8 sequence lane stream is missing");
        }
        const bool parent_anchor_already_ready =
            single_follower_stream_plan.anchor_on_parent_stream &&
            sequence == 0 &&
            (!parallel_project_qkv_z ||
             !handle->prune_parent_projection_waits_v4);
        if (parent_anchor_already_ready) {
          // Parent-stream ordering already covers the projection/input fork.
        } else if (parallel_project_qkv_z) {
          CheckAcl(aclrtStreamWaitEvent(sequence_stream, qkv_ready),
                   "aclrtStreamWaitEvent(GDN sequence QKV ready)");
          CheckAcl(aclrtStreamWaitEvent(sequence_stream, z_ready),
                   "aclrtStreamWaitEvent(GDN sequence Z ready)");
        } else {
          CheckAcl(aclrtStreamWaitEvent(sequence_stream, input_ready),
                   "aclrtStreamWaitEvent(GDN sequence input ready)");
        }
        aclrtEvent sequence_phase_done = nullptr;
        if (lane_local_packed_recurrent) {
          sequence_phase_done = SynchronizationEvent(
              aggregate, 1 + sequence,
              "aclrtCreateEvent(GDN lane recurrent ready v13)");
        } else if (lane_anchored_packed_recurrent && sequence == 1) {
          sequence_phase_done = SynchronizationEvent(
              aggregate, 1,
              "aclrtCreateEvent(GDN lane1 recurrent ready v14)");
        } else if (!lane_anchored_packed_recurrent) {
          sequence_phase_done = SynchronizationEvent(
              aggregate, 3 + sequence,
              "aclrtCreateEvent(GDN sequence done)");
        }

        std::array<statecentric_qwen38_tensor_v1, 10> sequence_inputs{};
        std::copy_n(execution->inputs, sequence_inputs.size(),
                    sequence_inputs.begin());
        const auto sequence_tokens = token_counts[sequence];
        const auto token_offset = token_offsets[sequence];
        sequence_inputs[0].data +=
            static_cast<std::uintptr_t>(token_offset * hidden_row_bytes);
        sequence_inputs[0].byte_length =
            static_cast<std::size_t>(sequence_tokens) * hidden_row_bytes;
        sequence_inputs[0].dimensions[0] = sequence_tokens;

        std::array<statecentric_qwen38_tensor_v1, 1> sequence_outputs{
            execution->outputs[0]};
        sequence_outputs[0].data +=
            static_cast<std::uintptr_t>(token_offset * hidden_row_bytes);
        sequence_outputs[0].byte_length =
            static_cast<std::size_t>(sequence_tokens) * hidden_row_bytes;
        sequence_outputs[0].dimensions[0] = sequence_tokens;

        std::array<statecentric_qwen38_state_region_v1, 2> sequence_state{
            execution->mutable_state[0], execution->mutable_state[1]};
        sequence_state[0].tensor.data +=
            static_cast<std::uintptr_t>(sequence * conv_state_bytes);
        sequence_state[0].tensor.byte_length = conv_state_bytes;
        sequence_state[0].tensor.dimensions[0] = 1;
        sequence_state[1].tensor.data +=
            static_cast<std::uintptr_t>(sequence * recurrent_state_bytes);
        sequence_state[1].tensor.byte_length = recurrent_state_bytes;
        sequence_state[1].tensor.dimensions[0] = 1;

        auto sequence_execution = *execution;
        sequence_execution.token_count = sequence_tokens;
        sequence_execution.workspace = 0;
        sequence_execution.workspace_length = 0;
        sequence_execution.stream =
            reinterpret_cast<std::uintptr_t>(sequence_stream);
        sequence_execution.inputs = sequence_inputs.data();
        sequence_execution.outputs = sequence_outputs.data();
        sequence_execution.mutable_state = sequence_state.data();
        statecentric_qwen38_precomputed_projection_v1
            sequence_precomputed{};
        if (batch_project_qkv_z || lane_packed_recurrent) {
          const auto qkv_row_bytes =
              static_cast<std::size_t>(kQkv) * sizeof(std::uint16_t);
          const auto z_row_bytes =
              static_cast<std::size_t>(kValue) * sizeof(std::uint16_t);
          sequence_precomputed.magic = kPrecomputedProjectionMagic;
          sequence_precomputed.tokens = sequence_tokens;
          sequence_precomputed.qkv =
              static_cast<std::uint8_t*>(batched_qkv) +
              token_offset * qkv_row_bytes;
          sequence_precomputed.z = static_cast<std::uint8_t*>(batched_z) +
                                   token_offset * z_row_bytes;
          sequence_precomputed.use_precomputed_projection =
              batch_project_qkv_z ? 1 : 0;
          if (lane_packed_recurrent) {
            const auto query_norm_row_bytes =
                static_cast<std::size_t>(kQueryHeads * kHeadDim) *
                sizeof(std::uint16_t);
            const auto beta_row_bytes =
                static_cast<std::size_t>(kGate) * sizeof(std::uint16_t);
            const auto gate_row_bytes =
                static_cast<std::size_t>(kGate) * sizeof(float);
            sequence_precomputed.query_norm =
                static_cast<std::uint8_t*>(batched_query_norm) +
                token_offset * query_norm_row_bytes;
            sequence_precomputed.key_norm =
                static_cast<std::uint8_t*>(batched_key_norm) +
                token_offset * query_norm_row_bytes;
            sequence_precomputed.value =
                static_cast<std::uint8_t*>(batched_value) +
                token_offset * z_row_bytes;
            sequence_precomputed.beta =
                static_cast<std::uint8_t*>(batched_beta) +
                token_offset * beta_row_bytes;
            sequence_precomputed.gate =
                static_cast<std::uint8_t*>(batched_gate) +
                token_offset * gate_row_bytes;
            sequence_precomputed.defer_recurrent = 1;
          }
          if (batch_output_tail) {
            sequence_precomputed.recurrent_output =
                static_cast<std::uint8_t*>(batched_recurrent_output) +
                token_offset * z_row_bytes;
            sequence_precomputed.defer_output_tail = 1;
          }
          if (batch_output_projection_only) {
            sequence_precomputed.gated_output =
                static_cast<std::uint8_t*>(batched_gated_output) +
                token_offset * z_row_bytes;
            sequence_precomputed.defer_output_projection = 1;
          }
          sequence_execution.workspace = reinterpret_cast<std::uintptr_t>(
              &sequence_precomputed);
          sequence_execution.workspace_length = sizeof(sequence_precomputed);
        }
        statecentric_qwen38_event_v1* sequence_event = nullptr;
        statecentric_qwen38_error_v1* sequence_error = nullptr;
        const auto status = Submit(handle, &sequence_execution,
                                   deadline_monotonic_ns, &sequence_event,
                                   &sequence_error);
        if (status != STATECENTRIC_QWEN38_OK || sequence_event == nullptr) {
          ReleaseSequenceAggregate(handle, aggregate, true);
          if (out_error != nullptr) {
            *out_error = sequence_error;
          } else if (sequence_error != nullptr) {
            delete reinterpret_cast<OwnedError*>(sequence_error);
          }
          return status == STATECENTRIC_QWEN38_OK
                     ? Fail(
                           STATECENTRIC_QWEN38_INTERNAL,
                           "Qwen3.8 sequence-normalized GDN returned no event",
                           out_error)
                     : status;
        }
        aggregate->sequence_events.push_back(sequence_event);
        if (lane_anchored_packed_recurrent) {
          if (sequence == 1) {
            CheckAcl(aclrtRecordEvent(sequence_phase_done, sequence_stream),
                     "aclrtRecordEvent(GDN lane1 recurrent ready v14)");
            CheckAcl(aclrtStreamWaitEvent(anchor_stream, sequence_phase_done),
                     "aclrtStreamWaitEvent(GDN lane1 recurrent ready v14)");
          }
        } else {
          CheckAcl(aclrtRecordEvent(sequence_phase_done, sequence_stream),
                   lane_local_packed_recurrent
                       ? "aclrtRecordEvent(GDN lane recurrent ready v13)"
                       : "aclrtRecordEvent(GDN sequence done)");
          CheckAcl(aclrtStreamWaitEvent(aggregate->stream, sequence_phase_done),
                   lane_local_packed_recurrent
                       ? "aclrtStreamWaitEvent(GDN lane recurrent ready v13)"
                       : "aclrtStreamWaitEvent(GDN sequence done)");
        }
      }
      if (lane_local_packed_recurrent) {
        PackedDecodeRecurrentV12(
            handle, batched_query_norm, batched_key_norm, batched_value,
            batched_beta, batched_gate,
            reinterpret_cast<void*>(recurrent_state.tensor.data),
            batched_recurrent_output, aggregate->stream, aggregate);
        const auto packed_ready = SynchronizationEvent(
            aggregate, 3,
            "aclrtCreateEvent(GDN packed recurrent ready v13)");
        CheckAcl(aclrtRecordEvent(packed_ready, aggregate->stream),
                 "aclrtRecordEvent(GDN packed recurrent ready v13)");
        const auto z_row_bytes =
            static_cast<std::size_t>(kValue) * sizeof(std::uint16_t);
        for (std::size_t sequence = 0; sequence < state_count; ++sequence) {
          const auto lane = stream_plan->lane_by_sequence[sequence];
          const auto sequence_stream = handle->sequence_streams.at(lane);
          const auto token_offset = token_offsets[sequence];
          const auto lane_done = SynchronizationEvent(
              aggregate, 4 + sequence,
              "aclrtCreateEvent(GDN lane output done v13)");
          CheckAcl(aclrtStreamWaitEvent(sequence_stream, packed_ready),
                   "aclrtStreamWaitEvent(GDN packed recurrent ready v13)");
          LaneGdnOutputTailV13(
              static_cast<std::uint8_t*>(batched_recurrent_output) +
                  token_offset * z_row_bytes,
              static_cast<std::uint8_t*>(batched_z) +
                  token_offset * z_row_bytes,
              reinterpret_cast<void*>(input[8].data),
              reinterpret_cast<void*>(input[9].data),
              reinterpret_cast<std::uint8_t*>(
                  reinterpret_cast<void*>(output.data)) +
                  token_offset * hidden_row_bytes,
              lane_local_packed_decode_plan.tokens_per_sequence, kValueHeads,
              kValue, kHidden, nullptr, sequence_stream,
              aggregate->sequence_events.at(sequence));
          CheckAcl(aclrtRecordEvent(lane_done, sequence_stream),
                   "aclrtRecordEvent(GDN lane output done v13)");
          CheckAcl(aclrtStreamWaitEvent(aggregate->stream, lane_done),
                   "aclrtStreamWaitEvent(GDN lane output done v13)");
        }
      } else if (lane_anchored_packed_recurrent) {
        PackedDecodeRecurrentV12(
            handle, batched_query_norm, batched_key_norm, batched_value,
            batched_beta, batched_gate,
            reinterpret_cast<void*>(recurrent_state.tensor.data),
            batched_recurrent_output, anchor_stream, aggregate);
        if (parent_local_decode_tail) {
          BatchGdnOutputTail(
              batched_recurrent_output, batched_z,
              reinterpret_cast<void*>(input[8].data),
              reinterpret_cast<void*>(input[9].data),
              reinterpret_cast<void*>(output.data),
              lane_anchored_packed_decode_plan.batch,
              lane_anchored_packed_decode_plan.tokens_per_sequence,
              kValueHeads, kValue, aggregate->stream, aggregate);
        } else {
          const auto packed_ready = SynchronizationEvent(
              aggregate, 2,
              "aclrtCreateEvent(GDN anchored packed recurrent ready v14)");
          CheckAcl(aclrtRecordEvent(packed_ready, anchor_stream),
                   "aclrtRecordEvent(GDN anchored packed recurrent ready v14)");
          CheckAcl(
              aclrtStreamWaitEvent(follower_stream, packed_ready),
              "aclrtStreamWaitEvent(GDN anchored packed recurrent ready v14)");
          const auto z_row_bytes =
              static_cast<std::size_t>(kValue) * sizeof(std::uint16_t);
          for (std::size_t sequence = 0; sequence < state_count; ++sequence) {
            const auto lane = stream_plan->lane_by_sequence[sequence];
            const auto sequence_stream =
                single_follower_stream_plan.anchor_on_parent_stream &&
                        sequence == 0
                    ? aggregate->stream
                    : handle->sequence_streams.at(lane);
            const auto token_offset = token_offsets[sequence];
            LaneGdnOutputTailV13(
                static_cast<std::uint8_t*>(batched_recurrent_output) +
                    token_offset * z_row_bytes,
                static_cast<std::uint8_t*>(batched_z) +
                    token_offset * z_row_bytes,
                reinterpret_cast<void*>(input[8].data),
                reinterpret_cast<void*>(input[9].data),
                reinterpret_cast<std::uint8_t*>(
                    reinterpret_cast<void*>(output.data)) +
                    token_offset * hidden_row_bytes,
                lane_anchored_packed_decode_plan.tokens_per_sequence,
                kValueHeads, kValue, kHidden,
                decode_flat_output_projection
                    ? static_cast<std::uint8_t*>(batched_gated_output) +
                          token_offset * z_row_bytes
                    : nullptr,
                sequence_stream,
                aggregate->sequence_events.at(sequence));
            if (!single_follower_stream_plan.anchor_on_parent_stream ||
                sequence != 0) {
              const auto lane_done = SynchronizationEvent(
                  aggregate, 3 + sequence,
                  "aclrtCreateEvent(GDN anchored lane output done v14)");
              CheckAcl(aclrtRecordEvent(lane_done, sequence_stream),
                       "aclrtRecordEvent(GDN anchored lane output done v14)");
              CheckAcl(
                  aclrtStreamWaitEvent(aggregate->stream, lane_done),
                  "aclrtStreamWaitEvent(GDN anchored lane output done v14)");
            }
          }
          if (decode_flat_output_projection) {
            FlatMatmulSharedWeight(
                batched_gated_output, reinterpret_cast<void*>(input[9].data),
                reinterpret_cast<void*>(output.data),
                static_cast<std::int64_t>(tokens), kValue, kHidden,
                aggregate->stream, aggregate,
                "aclnnMatmul(GDN flattened decode output v18)");
          }
        }
      } else if (batch_output_tail) {
        BatchGdnOutputTail(
            batched_recurrent_output, batched_z,
            reinterpret_cast<void*>(input[8].data),
            reinterpret_cast<void*>(input[9].data),
            reinterpret_cast<void*>(output.data), tail_plan.batch,
            tail_plan.tokens_per_sequence, kValueHeads, kValue,
            aggregate->stream, aggregate);
      } else if (batch_output_projection_only) {
        FlatMatmulSharedWeight(
            batched_gated_output, reinterpret_cast<void*>(input[9].data),
            reinterpret_cast<void*>(output.data),
            static_cast<std::int64_t>(tokens), kValue, kHidden,
            aggregate->stream, aggregate,
            "aclnnMatmul(GDN flattened dual-sequence output v11)");
      }
    } catch (const std::exception& error) {
      ReleaseSequenceAggregate(handle, aggregate, true);
      return Fail(STATECENTRIC_QWEN38_INTERNAL,
                  std::string("Qwen3.8 sequence stream submission failed: ") +
                      error.what(),
                  out_error);
    }
    *out_event = aggregate;
    return STATECENTRIC_QWEN38_OK;
  }
  auto* event = new (std::nothrow) statecentric_qwen38_event_v1;
  if (event == nullptr) {
    return Fail(STATECENTRIC_QWEN38_INTERNAL,
                "cannot allocate Qwen3.8 GDN event", out_error);
  }
  event->stream = reinterpret_cast<aclrtStream>(execution->stream);
  event->submitted_ns = started;
  try {
#ifdef STATECENTRIC_GDN_PREFILL_TENSOR_POOL_V1
    BindPrefillTensorPool(handle, event, tokens, precomputed != nullptr);
#endif
    EnsureHandleConstants(handle);
    const auto bf16 = sizeof(std::uint16_t);
    auto normalized = MakeTensor({static_cast<std::int64_t>(tokens), kHidden},
                                 {kHidden, 1},
                                 {static_cast<std::int64_t>(tokens), kHidden},
                                 ACL_BF16,
                                 reinterpret_cast<void*>(input[0].data));
    auto weight = [&](std::size_t index, std::int64_t rows,
                      std::int64_t columns) {
      return MakeTensor({rows, columns}, {1, rows}, {columns, rows}, ACL_BF16,
                        reinterpret_cast<void*>(input[index].data));
    };
    auto matrix = [&](void* data, std::int64_t columns) {
      return MakeTensor({static_cast<std::int64_t>(tokens), columns},
                        {columns, 1},
                        {static_cast<std::int64_t>(tokens), columns}, ACL_BF16,
                        data);
    };
    const bool use_precomputed_projection =
        precomputed != nullptr && precomputed->use_precomputed_projection == 1;
    void* qkv_data =
        precomputed != nullptr && precomputed->qkv != nullptr
            ? precomputed->qkv
            : Allocate(event, tokens * kQkv * bf16);
    void* z_data =
        precomputed != nullptr && precomputed->z != nullptr
            ? precomputed->z
            : Allocate(event, tokens * kValue * bf16);
    void* a_data = Allocate(event, tokens * kGate * bf16);
    void* b_data = Allocate(event, tokens * kGate * bf16);
    auto qkv = matrix(qkv_data, kQkv);
    auto z = matrix(z_data, kValue);
    auto a = matrix(a_data, kGate);
    auto b = matrix(b_data, kGate);
    auto qkv_weight = weight(1, kHidden, kQkv);
    auto z_weight = weight(2, kHidden, kValue);
    auto a_weight = weight(3, kHidden, kGate);
    auto b_weight = weight(4, kHidden, kGate);
    if (!use_precomputed_projection) {
      Matmul(normalized, qkv_weight, qkv, event->stream, event,
             "aclnnMatmul(GDN QKV)");
      Matmul(normalized, z_weight, z, event->stream, event,
             "aclnnMatmul(GDN Z)");
    }
    Matmul(normalized, a_weight, a, event->stream, event,
           "aclnnMatmul(GDN A)");
    Matmul(normalized, b_weight, b, event->stream, event,
           "aclnnMatmul(GDN B)");

    void* conv_output_data = Allocate(event, tokens * kQkv * bf16);
    auto conv_weight = MakeTensor(
        {4, kQkv}, {kQkv, 1}, {4, kQkv}, ACL_BF16,
        reinterpret_cast<void*>(input[5].data));
    auto conv_state_tensor = MakeTensor(
        {static_cast<std::int64_t>(state_count), kConvHistory, kQkv},
        {kConvHistory * kQkv, kQkv, 1},
        {static_cast<std::int64_t>(state_count), kConvHistory, kQkv}, ACL_BF16,
        reinterpret_cast<void*>(conv_state.tensor.data));
    auto conv_output = matrix(conv_output_data, kQkv);
    std::vector<std::int64_t> query_starts;
    std::vector<std::int64_t> cache_indices;
    query_starts.reserve(state_count + 1);
    cache_indices.reserve(state_count);
    if (multistate) {
      query_starts.push_back(0);
      for (std::uint64_t index = 0; index < state_count; ++index) {
        query_starts.push_back(static_cast<std::int64_t>(
            token_offsets[index] + token_counts[index]));
        cache_indices.push_back(static_cast<std::int64_t>(index));
      }
    } else {
      query_starts = {0, static_cast<std::int64_t>(tokens)};
      cache_indices = {0};
    }
    std::vector<std::int64_t> initial_mode(state_count, 0);
    auto query_starts_tensor = CopyInt64MetadataTensor(
        event, query_starts, "aclrtMemcpy(GDN conv query starts)");
    auto cache_indices_tensor = CopyInt64MetadataTensor(
        event, cache_indices, "aclrtMemcpy(GDN conv cache indices)");
    Tensor initial_state_tensor;
    if (execution->context_tokens == 0) {
      initial_state_tensor = CopyInt64MetadataTensor(
          event, initial_mode, "aclrtMemcpy(GDN conv initial-state mode)");
    }
    std::uint64_t conv_bytes = 0;
    aclOpExecutor* conv_executor = nullptr;
    CheckNn(aclnnCausalConv1dGetWorkspaceSize(
                qkv.get(), conv_weight.get(), nullptr, conv_state_tensor.get(),
                query_starts_tensor.get(), cache_indices_tensor.get(),
                initial_state_tensor.get(), nullptr, 1, -1,
                execution->context_tokens == 0 ? 0 : 1, conv_output.get(),
                &conv_bytes, &conv_executor),
            "aclnnCausalConv1dGetWorkspaceSize(GDN)");
    void* conv_workspace = GDN_WORKSPACE(event, conv_bytes, kConvolutionWorkspace);
    CheckNn(aclnnCausalConv1d(conv_workspace, conv_bytes, conv_executor,
                              event->stream),
            "aclnnCausalConv1d(GDN)");

    const auto query_elements = tokens * kQueryHeads * kHeadDim;
    const auto value_elements = tokens * kValueHeads * kHeadDim;
    const bool chunk_path =
        tokens >= kChunkKernelMinimumTokens && tokens <= kChunkSize &&
        !multistate && !handle->recurrent_prefill_always_recurrent;
    void* query_data = prefill_strided_qkv_v4
                           ? nullptr
                           : Allocate(event, query_elements * bf16);
    void* key_data = prefill_strided_qkv_v4
                         ? nullptr
                         : Allocate(event, query_elements * bf16);
    void* value_data =
        precomputed != nullptr && precomputed->defer_recurrent == 1
            ? precomputed->value
            : (prefill_strided_qkv_v4 && !chunk_path
                   ? nullptr
                   : Allocate(event, value_elements * bf16));
    const auto query_bytes = static_cast<std::size_t>(kQueryHeads * kHeadDim) * bf16;
    const auto value_bytes = static_cast<std::size_t>(kValueHeads * kHeadDim) * bf16;
    if (prefill_split_v2) {
      SplitQkv(conv_output_data, query_data, key_data, value_data, tokens,
               kQkv, kQueryHeads * kHeadDim, kValueHeads * kHeadDim,
               event->stream, event);
    } else if (prefill_split_v3) {
      SplitQkvRepeatable(handle, conv_output_data, query_data, key_data,
                         value_data, tokens, kQkv, kQueryHeads * kHeadDim,
                         kValueHeads * kHeadDim, event->stream, event);
    } else if (prefill_strided_qkv_v4) {
      if (chunk_path) {
        for (std::uint64_t token = 0; token < tokens; ++token) {
          auto* source = static_cast<std::uint8_t*>(conv_output_data) +
                         token * kQkv * bf16 + 2 * query_bytes;
          CheckAcl(aclrtMemcpyAsync(
                       static_cast<std::uint8_t*>(value_data) +
                           token * value_bytes,
                       value_bytes, source, value_bytes,
                       ACL_MEMCPY_DEVICE_TO_DEVICE, event->stream),
                   "aclrtMemcpyAsync(GDN strided chunk value fallback)");
        }
      }
    } else if (prefill_post_conv_pack_v5) {
      PostConvPack(handle, static_cast<std::uint32_t>(tokens),
                   conv_output_data, query_data, key_data, value_data,
                   event->stream);
    } else {
      for (std::uint64_t token = 0; token < tokens; ++token) {
        auto* source = static_cast<std::uint8_t*>(conv_output_data) +
                       token * kQkv * bf16;
        CheckAcl(aclrtMemcpyAsync(static_cast<std::uint8_t*>(query_data) +
                                      token * query_bytes,
                                  query_bytes, source, query_bytes,
                                  ACL_MEMCPY_DEVICE_TO_DEVICE, event->stream),
                 "aclrtMemcpyAsync(GDN query split)");
        CheckAcl(aclrtMemcpyAsync(static_cast<std::uint8_t*>(key_data) +
                                      token * query_bytes,
                                  query_bytes, source + query_bytes,
                                  query_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                  event->stream),
                 "aclrtMemcpyAsync(GDN key split)");
        CheckAcl(aclrtMemcpyAsync(static_cast<std::uint8_t*>(value_data) +
                                      token * value_bytes,
                                  value_bytes, source + 2 * query_bytes,
                                  value_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                  event->stream),
                 "aclrtMemcpyAsync(GDN value split)");
      }
    }

    auto l2_gamma_tensor = MakeTensor({kHeadDim}, {1}, {kHeadDim}, ACL_BF16,
                                      handle->l2_gamma);
    const auto token_rows = static_cast<std::int64_t>(tokens);
    const auto query_rows = token_rows * kQueryHeads;
    auto query = prefill_strided_qkv_v4
                     ? MakeTensor({token_rows, kQueryHeads, kHeadDim},
                                  {kQkv, kHeadDim, 1}, {token_rows, kQkv},
                                  ACL_BF16, conv_output_data)
                     : MakeTensor({query_rows, kHeadDim}, {kHeadDim, 1},
                                  {query_rows, kHeadDim}, ACL_BF16,
                                  query_data);
    auto key = prefill_strided_qkv_v4
                   ? MakeTensor({token_rows, kQueryHeads, kHeadDim},
                                {kQkv, kHeadDim, 1}, {token_rows, kQkv},
                                ACL_BF16, conv_output_data,
                                kQueryHeads * kHeadDim)
                   : MakeTensor({query_rows, kHeadDim}, {kHeadDim, 1},
                                {query_rows, kHeadDim}, ACL_BF16, key_data);
    void* query_norm_data =
        precomputed != nullptr && precomputed->defer_recurrent == 1
            ? precomputed->query_norm
            : Allocate(event, query_elements * bf16);
    void* key_norm_data =
        precomputed != nullptr && precomputed->defer_recurrent == 1
            ? precomputed->key_norm
            : Allocate(event, query_elements * bf16);
    auto query_norm = prefill_strided_qkv_v4
                          ? MakeTensor(
                                {token_rows, kQueryHeads, kHeadDim},
                                {kQueryHeads * kHeadDim, kHeadDim, 1},
                                {token_rows, kQueryHeads, kHeadDim}, ACL_BF16,
                                query_norm_data)
                          : MakeTensor({query_rows, kHeadDim}, {kHeadDim, 1},
                                       {query_rows, kHeadDim}, ACL_BF16,
                                       query_norm_data);
    auto key_norm = prefill_strided_qkv_v4
                        ? MakeTensor(
                              {token_rows, kQueryHeads, kHeadDim},
                              {kQueryHeads * kHeadDim, kHeadDim, 1},
                              {token_rows, kQueryHeads, kHeadDim}, ACL_BF16,
                              key_norm_data)
                        : MakeTensor({query_rows, kHeadDim}, {kHeadDim, 1},
                                     {query_rows, kHeadDim}, ACL_BF16,
                                     key_norm_data);
    if (prefill_strided_qkv_v4) {
      RmsNormHeads(query, l2_gamma_tensor, query_norm, token_rows,
                   kQueryHeads,
                   1.0e-6 / static_cast<double>(kHeadDim), event->stream,
                   event, "aclnnRmsNorm(GDN strided query)");
      RmsNormHeads(key, l2_gamma_tensor, key_norm, token_rows, kQueryHeads,
                   1.0e-6 / static_cast<double>(kHeadDim), event->stream,
                   event, "aclnnRmsNorm(GDN strided key)");
    } else {
      RmsNorm(query, l2_gamma_tensor, query_norm, query_rows,
              1.0e-6 / static_cast<double>(kHeadDim), event->stream, event,
              "aclnnRmsNorm(GDN query)");
      RmsNorm(key, l2_gamma_tensor, key_norm, query_rows,
              1.0e-6 / static_cast<double>(kHeadDim), event->stream, event,
              "aclnnRmsNorm(GDN key)");
    }

    void* g_data =
        precomputed != nullptr && precomputed->defer_recurrent == 1
            ? precomputed->gate
            : Allocate(event, tokens * kGate * sizeof(float));
    void* beta_data =
        precomputed != nullptr && precomputed->defer_recurrent == 1
            ? precomputed->beta
            : Allocate(event, tokens * kGate * bf16);
    auto a_log = MakeTensor({kGate}, {1}, {kGate}, ACL_FLOAT,
                            reinterpret_cast<void*>(input[6].data));
    auto dt_bias = MakeTensor({kGate}, {1}, {kGate}, ACL_FLOAT,
                              reinterpret_cast<void*>(input[7].data));
    auto g = MakeTensor({1, static_cast<std::int64_t>(tokens), kGate},
                        {static_cast<std::int64_t>(tokens) * kGate, kGate, 1},
                        {1, static_cast<std::int64_t>(tokens), kGate}, ACL_FLOAT,
                        g_data);
    auto beta = MakeTensor(
        {1, static_cast<std::int64_t>(tokens), kGate},
        {static_cast<std::int64_t>(tokens) * kGate, kGate, 1},
        {1, static_cast<std::int64_t>(tokens), kGate}, ACL_BF16, beta_data);
    std::uint64_t gating_bytes = 0;
    aclOpExecutor* gating_executor = nullptr;
    CheckNn(aclnnFusedGdnGatingGetWorkspaceSize(
                a_log.get(), a.get(), b.get(), dt_bias.get(), 1.0F, 20.0F,
                g.get(), beta.get(), &gating_bytes, &gating_executor),
            "aclnnFusedGdnGatingGetWorkspaceSize(GDN)");
    void* gating_workspace = GDN_WORKSPACE(event, gating_bytes, kGatingWorkspace);
    CheckNn(aclnnFusedGdnGating(gating_workspace, gating_bytes,
                                gating_executor, event->stream),
            "aclnnFusedGdnGating(GDN)");

    if (precomputed != nullptr && precomputed->defer_recurrent == 1) {
      *out_event = event;
      return STATECENTRIC_QWEN38_OK;
    }

    const bool defer_output_tail =
        precomputed != nullptr && precomputed->defer_output_tail == 1;
    void* recurrent_output_data = defer_output_tail
                                      ? precomputed->recurrent_output
                                      : Allocate(event, value_elements * bf16);
    if (chunk_path) {
      ChunkGdn(handle, static_cast<std::uint32_t>(tokens), query_norm_data,
               key_norm_data, value_data, beta_data, g_data,
               reinterpret_cast<void*>(recurrent_state.tensor.data),
               recurrent_output_data, event->stream, event);
    } else {
      const auto query_token_bytes =
          static_cast<std::size_t>(kQueryHeads * kHeadDim) * bf16;
      const auto value_token_bytes =
          static_cast<std::size_t>(kValueHeads * kHeadDim) * bf16;
      const auto recurrent_state_bytes =
          static_cast<std::size_t>(kValueHeads * kHeadDim * kHeadDim) *
          sizeof(float);
      const auto run_chunk_sequence =
          [&](std::uint64_t sequence, std::uint64_t sequence_offset,
              std::uint64_t sequence_tokens) {
            ChunkGdn(
                handle, static_cast<std::uint32_t>(sequence_tokens),
                static_cast<std::uint8_t*>(query_norm_data) +
                    sequence_offset * query_token_bytes,
                static_cast<std::uint8_t*>(key_norm_data) +
                    sequence_offset * query_token_bytes,
                static_cast<std::uint8_t*>(value_data) +
                    sequence_offset * value_token_bytes,
                static_cast<std::uint8_t*>(beta_data) +
                    sequence_offset * static_cast<std::size_t>(kGate) * bf16,
                static_cast<std::uint8_t*>(g_data) +
                    sequence_offset * static_cast<std::size_t>(kGate) *
                        sizeof(float),
                static_cast<std::uint8_t*>(
                    reinterpret_cast<void*>(recurrent_state.tensor.data)) +
                    sequence * recurrent_state_bytes,
                static_cast<std::uint8_t*>(recurrent_output_data) +
                    sequence_offset * value_token_bytes,
                event->stream, event);
          };
      const auto run_sequence = [&](std::uint64_t sequence,
                                    std::uint64_t sequence_offset,
                                    std::uint64_t sequence_tokens) {
        auto recurrent = MakeTensor(
            {1, kValueHeads, kHeadDim, kHeadDim},
            {kValueHeads * kHeadDim * kHeadDim, kHeadDim * kHeadDim,
             kHeadDim, 1},
            {1, kValueHeads, kHeadDim, kHeadDim}, ACL_FLOAT,
            static_cast<std::uint8_t*>(
                reinterpret_cast<void*>(recurrent_state.tensor.data)) +
                sequence * recurrent_state_bytes);
        for (std::uint64_t local_offset = 0; local_offset < sequence_tokens;
             local_offset += kRecurrentKernelTokens) {
        const auto offset = sequence_offset + local_offset;
        const auto slice_tokens =
            static_cast<std::uint32_t>(std::min<std::uint64_t>(
                kRecurrentKernelTokens, sequence_tokens - local_offset));
        const auto slice = static_cast<std::int64_t>(slice_tokens);
        const auto constant_index = static_cast<std::size_t>(slice_tokens - 1);
        auto query_recurrent = MakeTensor(
            {slice, kQueryHeads, kHeadDim},
            {kQueryHeads * kHeadDim, kHeadDim, 1},
            {slice, kQueryHeads, kHeadDim}, ACL_BF16,
            static_cast<std::uint8_t*>(query_norm_data) +
                offset * query_token_bytes);
        auto key_recurrent = MakeTensor(
            {slice, kQueryHeads, kHeadDim},
            {kQueryHeads * kHeadDim, kHeadDim, 1},
            {slice, kQueryHeads, kHeadDim}, ACL_BF16,
            static_cast<std::uint8_t*>(key_norm_data) +
                offset * query_token_bytes);
        auto value = prefill_strided_qkv_v4
                         ? MakeTensor(
                               {slice, kValueHeads, kHeadDim},
                               {kQkv, kHeadDim, 1}, {token_rows, kQkv},
                               ACL_BF16, conv_output_data,
                               static_cast<std::int64_t>(offset * kQkv) +
                                   2 * kQueryHeads * kHeadDim)
                         : MakeTensor(
                               {slice, kValueHeads, kHeadDim},
                               {kValueHeads * kHeadDim, kHeadDim, 1},
                               {slice, kValueHeads, kHeadDim}, ACL_BF16,
                               static_cast<std::uint8_t*>(value_data) +
                                   offset * value_token_bytes);
        auto beta_recurrent = MakeTensor(
            {slice, kGate}, {kGate, 1}, {slice, kGate}, ACL_BF16,
            static_cast<std::uint8_t*>(beta_data) +
                offset * static_cast<std::size_t>(kGate) * bf16);
        auto g_recurrent = MakeTensor(
            {slice, kGate}, {kGate, 1}, {slice, kGate}, ACL_FLOAT,
            static_cast<std::uint8_t*>(g_data) +
                offset * static_cast<std::size_t>(kGate) * sizeof(float));
        void* actual_lengths_data = handle->actual_lengths[constant_index];
        void* state_indices_data = handle->state_indices[constant_index];
        auto actual_lengths_tensor = MakeTensor(
            {2}, {1}, {2}, ACL_INT32,
            actual_lengths_data);
        auto state_indices_tensor = MakeTensor(
            {slice}, {1}, {slice}, ACL_INT32, state_indices_data);
        auto recurrent_output = MakeTensor(
            {slice, kValueHeads, kHeadDim},
            {kValueHeads * kHeadDim, kHeadDim, 1},
            {slice, kValueHeads, kHeadDim}, ACL_BF16,
            static_cast<std::uint8_t*>(recurrent_output_data) +
                offset * value_token_bytes);
        std::uint64_t recurrent_bytes = 0;
        aclOpExecutor* recurrent_executor = nullptr;
        CheckNn(aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
                    query_recurrent.get(), key_recurrent.get(), value.get(),
                    beta_recurrent.get(), recurrent.get(),
                    actual_lengths_tensor.get(), state_indices_tensor.get(),
                    g_recurrent.get(), nullptr, nullptr,
                    std::pow(static_cast<float>(kHeadDim), -0.5F),
                    recurrent_output.get(), &recurrent_bytes,
                    &recurrent_executor),
                "aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(GDN slice)");
        void* recurrent_workspace = GDN_WORKSPACE(event, recurrent_bytes, kRecurrentWorkspace);
        CheckNn(aclnnRecurrentGatedDeltaRule(
                    recurrent_workspace, recurrent_bytes, recurrent_executor,
                    event->stream),
                "aclnnRecurrentGatedDeltaRule(GDN slice)");
        }
      };
      if (packed_decode_plan.packed_recurrent) {
        PackedDecodeRecurrentV12(
            handle, query_norm_data, key_norm_data, value_data, beta_data,
            g_data, reinterpret_cast<void*>(recurrent_state.tensor.data),
            recurrent_output_data, event->stream, event);
      } else if (multistate) {
        for (std::uint64_t sequence = 0; sequence < state_count; ++sequence) {
          const auto sequence_tokens = token_counts[sequence];
          if (statecentric::UsePackedSequenceChunkGdn(
                  multistate, prefill_post_conv_pack_v5,
                  handle->recurrent_prefill_always_recurrent,
                  sequence_tokens)) {
            run_chunk_sequence(sequence, token_offsets[sequence],
                               sequence_tokens);
          } else {
            run_sequence(sequence, token_offsets[sequence], sequence_tokens);
          }
        }
      } else {
        run_sequence(0, 0, tokens);
      }
    }

    if (defer_output_tail) {
      *out_event = event;
      return STATECENTRIC_QWEN38_OK;
    }

    const bool defer_output_projection =
        precomputed != nullptr && precomputed->defer_output_projection == 1;

    const auto value_rows = static_cast<std::int64_t>(tokens) * kValueHeads;
    auto recurrent_flat = MakeTensor({value_rows, kHeadDim}, {kHeadDim, 1},
                                     {value_rows, kHeadDim}, ACL_BF16,
                                     recurrent_output_data);
    auto output_gamma = MakeTensor({kHeadDim}, {1}, {kHeadDim}, ACL_BF16,
                                   reinterpret_cast<void*>(input[8].data));
    void* normalized_output_data = Allocate(event, value_elements * bf16);
    auto normalized_output = MakeTensor(
        {value_rows, kHeadDim}, {kHeadDim, 1}, {value_rows, kHeadDim}, ACL_BF16,
        normalized_output_data);
    RmsNorm(recurrent_flat, output_gamma, normalized_output, value_rows, 1.0e-6,
            event->stream, event, "aclnnRmsNorm(GDN output)");

    void* activated_data = Allocate(event, tokens * kValue * bf16);
    void* gated_data = defer_output_projection
                           ? precomputed->gated_output
                           : Allocate(event, tokens * kValue * bf16);
    auto activated = matrix(activated_data, kValue);
    auto normalized_value = matrix(normalized_output_data, kValue);
    auto gated = matrix(gated_data, kValue);
    std::uint64_t silu_bytes = 0;
    aclOpExecutor* silu_executor = nullptr;
    CheckNn(aclnnSiluGetWorkspaceSize(z.get(), activated.get(), &silu_bytes,
                                      &silu_executor),
            "aclnnSiluGetWorkspaceSize(GDN output)");
    void* silu_workspace = GDN_WORKSPACE(event, silu_bytes, kElementwiseWorkspace);
    CheckNn(aclnnSilu(silu_workspace, silu_bytes, silu_executor, event->stream),
            "aclnnSilu(GDN output)");
    std::uint64_t mul_bytes = 0;
    aclOpExecutor* mul_executor = nullptr;
    CheckNn(aclnnMulGetWorkspaceSize(activated.get(), normalized_value.get(),
                                     gated.get(), &mul_bytes, &mul_executor),
            "aclnnMulGetWorkspaceSize(GDN output)");
    void* mul_workspace = GDN_WORKSPACE(event, mul_bytes, kElementwiseWorkspace);
    CheckNn(aclnnMul(mul_workspace, mul_bytes, mul_executor, event->stream),
            "aclnnMul(GDN output)");
    if (defer_output_projection) {
      *out_event = event;
      return STATECENTRIC_QWEN38_OK;
    }
    auto out_weight = weight(9, kValue, kHidden);
    auto local_attention = MakeTensor(
        {static_cast<std::int64_t>(tokens), kHidden}, {kHidden, 1},
        {static_cast<std::int64_t>(tokens), kHidden}, ACL_BF16,
        reinterpret_cast<void*>(execution->outputs[0].data));
    Matmul(gated, out_weight, local_attention, event->stream, event,
           "aclnnMatmul(GDN output projection)");
  } catch (const std::exception& error) {
    event->complete = aclrtSynchronizeStream(event->stream) == ACL_SUCCESS;
    ReleaseSplitCaches(handle, event);
    FreeAllocations(event);
    delete event;
    return Fail(STATECENTRIC_QWEN38_INTERNAL,
                std::string("Qwen3.8 GDN submission failed: ") + error.what(),
                out_error);
  }
  *out_event = event;
  return STATECENTRIC_QWEN38_OK;
}

void FillCompletion(statecentric_qwen38_completion_v1* completion,
                    std::uint32_t phase, std::uint64_t submitted,
                    std::uint64_t completed) {
  *completion = {sizeof(*completion), STATECENTRIC_QWEN38_PROVIDER_ABI_V1,
                 phase, 0,
                 completed >= submitted ? completed - submitted : 0,
                 completed};
}

int32_t Poll(statecentric_qwen38_provider_handle_v1* handle,
             statecentric_qwen38_event_v1* event,
             statecentric_qwen38_completion_v1* completion,
             statecentric_qwen38_error_v1** out_error) {
  if (handle == nullptr || event == nullptr || completion == nullptr) {
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "invalid Qwen3.8 GDN event poll", out_error);
  }
  aclrtStreamStatus stream_status = ACL_STREAM_STATUS_RESERVED;
  const auto status = aclrtStreamQuery(event->stream, &stream_status);
  if (status != ACL_SUCCESS) {
    return Fail(STATECENTRIC_QWEN38_INTERNAL,
                "aclrtStreamQuery(GDN) failed with status " +
                    std::to_string(status),
                out_error);
  }
  const auto now = handle->host->monotonic_time_ns(handle->host->context);
  if (stream_status == ACL_STREAM_STATUS_COMPLETE) {
    event->complete = true;
    FillCompletion(completion, STATECENTRIC_QWEN38_EVENT_COMPLETE,
                   event->submitted_ns, now);
    return STATECENTRIC_QWEN38_OK;
  }
  FillCompletion(completion, STATECENTRIC_QWEN38_EVENT_PENDING,
                 event->submitted_ns, 0);
  return STATECENTRIC_QWEN38_NOT_READY;
}

int32_t Wait(statecentric_qwen38_provider_handle_v1* handle,
             statecentric_qwen38_event_v1* event,
             std::uint64_t deadline_monotonic_ns,
             statecentric_qwen38_completion_v1* completion,
             statecentric_qwen38_error_v1** out_error) {
  if (handle == nullptr || event == nullptr || completion == nullptr) {
    return Fail(STATECENTRIC_QWEN38_INVALID_ARGUMENT,
                "invalid Qwen3.8 GDN event wait", out_error);
  }
  const auto now = handle->host->monotonic_time_ns(handle->host->context);
  if (deadline_monotonic_ns <= now) {
    return Fail(STATECENTRIC_QWEN38_TIMEOUT,
                "Qwen3.8 GDN event deadline expired", out_error);
  }
  const auto remaining = deadline_monotonic_ns - now;
  const auto timeout_ms = std::min<std::uint64_t>(
      remaining / 1000000 + (remaining % 1000000 != 0 ? 1 : 0),
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()));
  const auto status = aclrtSynchronizeStreamWithTimeout(
      event->stream, static_cast<std::int32_t>(timeout_ms));
  if (status != ACL_SUCCESS) {
    return Fail(STATECENTRIC_QWEN38_TIMEOUT,
                "aclrtSynchronizeStreamWithTimeout(GDN) failed with status " +
                    std::to_string(status),
                out_error);
  }
  event->complete = true;
  const auto completed = handle->host->monotonic_time_ns(handle->host->context);
  FillCompletion(completion, STATECENTRIC_QWEN38_EVENT_COMPLETE,
                 event->submitted_ns, completed);
  return STATECENTRIC_QWEN38_OK;
}

int32_t Cancel(statecentric_qwen38_provider_handle_v1*,
               statecentric_qwen38_event_v1*,
               statecentric_qwen38_error_v1** out_error) {
  return Fail(STATECENTRIC_QWEN38_UNSUPPORTED,
              "submitted Qwen3.8 GDN work cannot be cancelled safely",
              out_error);
}

void ReleaseEvent(statecentric_qwen38_provider_handle_v1* handle,
                  statecentric_qwen38_event_v1* event) {
  if (event == nullptr) return;
  if (!event->complete) {
    event->complete = aclrtSynchronizeStream(event->stream) == ACL_SUCCESS;
  }
  if (!event->sequence_events.empty() ||
      !event->synchronization_events.empty()) {
    ReleaseSequenceAggregate(handle, event, false);
    return;
  }
  ReleaseSplitCaches(handle, event);
  FreeAllocations(event);
  DestroySynchronizationEvents(event);
  delete event;
}

void ReleaseError(statecentric_qwen38_error_v1* error) {
  delete reinterpret_cast<OwnedError*>(error);
}

}  // namespace

extern "C" int32_t statecentric_qwen38_provider_query_v1(
    std::uint32_t host_abi_version, std::uint32_t host_descriptor_size,
    statecentric_qwen38_provider_descriptor_v1* out_descriptor,
    statecentric_qwen38_error_v1** out_error) {
  if (host_abi_version != STATECENTRIC_QWEN38_PROVIDER_ABI_V1 ||
      host_descriptor_size < STATECENTRIC_QWEN38_PROVIDER_DESCRIPTOR_V1_MIN_SIZE ||
      out_descriptor == nullptr) {
    return Fail(STATECENTRIC_QWEN38_ABI_MISMATCH,
                "Qwen3.8 GDN provider query ABI mismatch", out_error);
  }
  const statecentric_qwen38_provider_descriptor_v1 descriptor{
      sizeof(statecentric_qwen38_provider_descriptor_v1),
      STATECENTRIC_QWEN38_PROVIDER_ABI_V1,
      "org.statecentric.qwen38.gdn",
      "1.0.0",
      "operator.qwen38.gdn",
      1,
      STATECENTRIC_QWEN38_STAGE_GDN,
      STATECENTRIC_QWEN38_PROVIDER_FLAG_NO_PYTHON_IPC_V1,
      Create,
      Destroy,
      Submit,
      Poll,
      Wait,
      Cancel,
      ReleaseEvent,
      ReleaseError,
      nullptr,
      nullptr,
      nullptr};
  std::memcpy(out_descriptor, &descriptor, sizeof(descriptor));
  return STATECENTRIC_QWEN38_OK;
}
