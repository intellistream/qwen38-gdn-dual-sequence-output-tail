#include <array>
#include <chrono>
#include <iomanip>
#include <numeric>

#define main statecentric_qwen38_dual_sequence_tail_equivalence_main
#include "qwen38_dual_sequence_tail_equivalence_probe.cpp"
#undef main

namespace {

constexpr std::int64_t kDecodeTokens = 1;
constexpr std::uint64_t kDecodeTailWarmups = 3;
constexpr std::uint64_t kDecodeTailSamples = 12;

struct DecodeTailTiming {
  std::string arm;
  std::vector<double> device_us;
  std::vector<double> host_us;
};

void EnqueueRmsNormOnStream(Runtime& runtime, const Tensor& input,
                            const Tensor& gamma, const Tensor& output,
                            const std::vector<std::int64_t>& rstd_shape,
                            aclrtStream stream) {
  std::size_t rstd_elements = 1;
  for (const auto dimension : rstd_shape) {
    rstd_elements *= static_cast<std::size_t>(dimension);
  }
  void* rstd_data = runtime.Allocate(rstd_elements * sizeof(float));
  auto rstd = MakeContiguousTensor(rstd_shape, ACL_FLOAT, rstd_data);
  std::uint64_t workspace_bytes = 0;
  aclOpExecutor* executor = nullptr;
  CheckNn(aclnnRmsNormGetWorkspaceSize(
              input.get(), gamma.get(), 1.0e-6, output.get(), rstd.get(),
              &workspace_bytes, &executor),
          "aclnnRmsNormGetWorkspaceSize(parent-local decode tail)");
  void* workspace = runtime.Allocate(workspace_bytes);
  CheckNn(aclnnRmsNorm(workspace, workspace_bytes, executor, stream),
          "aclnnRmsNorm(parent-local decode tail)");
}

void EnqueueSiluOnStream(Runtime& runtime, const Tensor& input,
                         const Tensor& output, aclrtStream stream) {
  std::uint64_t workspace_bytes = 0;
  aclOpExecutor* executor = nullptr;
  CheckNn(aclnnSiluGetWorkspaceSize(input.get(), output.get(),
                                    &workspace_bytes, &executor),
          "aclnnSiluGetWorkspaceSize(parent-local decode tail)");
  void* workspace = runtime.Allocate(workspace_bytes);
  CheckNn(aclnnSilu(workspace, workspace_bytes, executor, stream),
          "aclnnSilu(parent-local decode tail)");
}

void EnqueueMulOnStream(Runtime& runtime, const Tensor& left,
                        const Tensor& right, const Tensor& output,
                        aclrtStream stream) {
  std::uint64_t workspace_bytes = 0;
  aclOpExecutor* executor = nullptr;
  CheckNn(aclnnMulGetWorkspaceSize(left.get(), right.get(), output.get(),
                                   &workspace_bytes, &executor),
          "aclnnMulGetWorkspaceSize(parent-local decode tail)");
  void* workspace = runtime.Allocate(workspace_bytes);
  CheckNn(aclnnMul(workspace, workspace_bytes, executor, stream),
          "aclnnMul(parent-local decode tail)");
}

struct DecodeTailFixture {
  explicit DecodeTailFixture(std::int32_t device) : runtime(device) {
    CheckAcl(aclrtCreateStream(&follower), "aclrtCreateStream(follower)");
    CheckAcl(aclrtCreateEvent(&packed_ready),
             "aclrtCreateEvent(packed ready)");
    CheckAcl(aclrtCreateEvent(&follower_done),
             "aclrtCreateEvent(follower done)");

    const auto value_elements = static_cast<std::size_t>(kBatch * kValue);
    const auto output_elements = static_cast<std::size_t>(kBatch * kHidden);
    const auto recurrent_host = Generate(value_elements, 0x1931A55AU);
    const auto z_host = Generate(value_elements, 0x18F0110BU);
    const auto gamma_host = Generate(kHeadDim, 0x13572468U);
    const auto weight_host = Generate(
        static_cast<std::size_t>(kValue * kHidden), 0x24681357U);

    recurrent = runtime.ToDevice(recurrent_host.data(),
                                 recurrent_host.size() * sizeof(std::uint16_t));
    z = runtime.ToDevice(z_host.data(),
                         z_host.size() * sizeof(std::uint16_t));
    gamma = runtime.ToDevice(gamma_host.data(),
                             gamma_host.size() * sizeof(std::uint16_t));
    weight = runtime.ToDevice(weight_host.data(),
                              weight_host.size() * sizeof(std::uint16_t));

    follower_normalized =
        runtime.Allocate(value_elements * sizeof(std::uint16_t));
    follower_activated =
        runtime.Allocate(value_elements * sizeof(std::uint16_t));
    follower_gated = runtime.Allocate(value_elements * sizeof(std::uint16_t));
    follower_output =
        runtime.Allocate(output_elements * sizeof(std::uint16_t));
    parent_normalized =
        runtime.Allocate(value_elements * sizeof(std::uint16_t));
    parent_activated =
        runtime.Allocate(value_elements * sizeof(std::uint16_t));
    parent_gated = runtime.Allocate(value_elements * sizeof(std::uint16_t));
    parent_output = runtime.Allocate(output_elements * sizeof(std::uint16_t));

    gamma_tensor = MakeContiguousTensor({kHeadDim}, ACL_BF16, gamma);
    weight_tensor = MakeTensor({kValue, kHidden}, {1, kValue},
                               {kHidden, kValue}, ACL_BF16, weight);
    const auto value_bytes =
        static_cast<std::size_t>(kValue * sizeof(std::uint16_t));
    for (std::int64_t sequence = 0; sequence < kBatch; ++sequence) {
      const auto offset = static_cast<std::size_t>(sequence) * value_bytes;
      follower_recurrent[sequence] = MakeContiguousTensor(
          {kValueHeads, kHeadDim}, ACL_BF16,
          static_cast<std::uint8_t*>(recurrent) + offset);
      follower_normalized_heads[sequence] = MakeContiguousTensor(
          {kValueHeads, kHeadDim}, ACL_BF16,
          static_cast<std::uint8_t*>(follower_normalized) + offset);
      follower_normalized_values[sequence] = MakeContiguousTensor(
          {kDecodeTokens, kValue}, ACL_BF16,
          static_cast<std::uint8_t*>(follower_normalized) + offset);
      follower_z[sequence] = MakeContiguousTensor(
          {kDecodeTokens, kValue}, ACL_BF16,
          static_cast<std::uint8_t*>(z) + offset);
      follower_activated_values[sequence] = MakeContiguousTensor(
          {kDecodeTokens, kValue}, ACL_BF16,
          static_cast<std::uint8_t*>(follower_activated) + offset);
      follower_gated_values[sequence] = MakeContiguousTensor(
          {kDecodeTokens, kValue}, ACL_BF16,
          static_cast<std::uint8_t*>(follower_gated) + offset);
    }

    parent_recurrent = MakeContiguousTensor(
        {kBatch, kValueHeads, kHeadDim}, ACL_BF16, recurrent);
    parent_normalized_heads = MakeContiguousTensor(
        {kBatch, kValueHeads, kHeadDim}, ACL_BF16, parent_normalized);
    parent_normalized_values = MakeContiguousTensor(
        {kBatch, kDecodeTokens, kValue}, ACL_BF16, parent_normalized);
    parent_z = MakeContiguousTensor(
        {kBatch, kDecodeTokens, kValue}, ACL_BF16, z);
    parent_activated_values = MakeContiguousTensor(
        {kBatch, kDecodeTokens, kValue}, ACL_BF16, parent_activated);
    parent_gated_values = MakeContiguousTensor(
        {kBatch, kDecodeTokens, kValue}, ACL_BF16, parent_gated);
    follower_flat_gated = MakeContiguousTensor(
        {kBatch, kValue}, ACL_BF16, follower_gated);
    parent_flat_gated = MakeContiguousTensor(
        {kBatch, kValue}, ACL_BF16, parent_gated);
    follower_output_tensor = MakeContiguousTensor(
        {kBatch, kHidden}, ACL_BF16, follower_output);
    parent_output_tensor = MakeContiguousTensor(
        {kBatch, kHidden}, ACL_BF16, parent_output);
  }

  DecodeTailFixture(const DecodeTailFixture&) = delete;
  DecodeTailFixture& operator=(const DecodeTailFixture&) = delete;

  ~DecodeTailFixture() {
    if (follower != nullptr) (void)aclrtSynchronizeStream(follower);
    if (follower_done != nullptr) (void)aclrtDestroyEvent(follower_done);
    if (packed_ready != nullptr) (void)aclrtDestroyEvent(packed_ready);
    if (follower != nullptr) (void)aclrtDestroyStream(follower);
  }

  void LaunchV18FollowerTail() {
    CheckAcl(aclrtRecordEvent(packed_ready, runtime.stream()),
             "aclrtRecordEvent(V18 packed ready)");
    CheckAcl(aclrtStreamWaitEvent(follower, packed_ready),
             "aclrtStreamWaitEvent(V18 packed ready)");
    LaunchLane(0, runtime.stream());
    LaunchLane(1, follower);
    CheckAcl(aclrtRecordEvent(follower_done, follower),
             "aclrtRecordEvent(V18 follower done)");
    CheckAcl(aclrtStreamWaitEvent(runtime.stream(), follower_done),
             "aclrtStreamWaitEvent(V18 follower done)");
    EnqueueMatmul(runtime, follower_flat_gated, weight_tensor,
                  follower_output_tensor);
  }

  void LaunchParentLocalTail() {
    EnqueueRmsNorm(runtime, parent_recurrent, gamma_tensor,
                   parent_normalized_heads, {kBatch, kValueHeads, 1});
    EnqueueSilu(runtime, parent_z, parent_activated_values);
    EnqueueMul(runtime, parent_activated_values, parent_normalized_values,
               parent_gated_values);
    EnqueueMatmul(runtime, parent_flat_gated, weight_tensor,
                  parent_output_tensor);
  }

  void Synchronize() {
    runtime.Synchronize();
    CheckAcl(aclrtSynchronizeStream(follower),
             "aclrtSynchronizeStream(follower)");
  }

  void LaunchLane(std::size_t sequence, aclrtStream stream) {
    EnqueueRmsNormOnStream(runtime, follower_recurrent[sequence], gamma_tensor,
                           follower_normalized_heads[sequence],
                           {kValueHeads, 1}, stream);
    EnqueueSiluOnStream(runtime, follower_z[sequence],
                        follower_activated_values[sequence], stream);
    EnqueueMulOnStream(runtime, follower_activated_values[sequence],
                       follower_normalized_values[sequence],
                       follower_gated_values[sequence], stream);
  }

  Runtime runtime;
  aclrtStream follower = nullptr;
  aclrtEvent packed_ready = nullptr;
  aclrtEvent follower_done = nullptr;
  void* recurrent = nullptr;
  void* z = nullptr;
  void* gamma = nullptr;
  void* weight = nullptr;
  void* follower_normalized = nullptr;
  void* follower_activated = nullptr;
  void* follower_gated = nullptr;
  void* follower_output = nullptr;
  void* parent_normalized = nullptr;
  void* parent_activated = nullptr;
  void* parent_gated = nullptr;
  void* parent_output = nullptr;
  Tensor gamma_tensor;
  Tensor weight_tensor;
  std::array<Tensor, kBatch> follower_recurrent;
  std::array<Tensor, kBatch> follower_normalized_heads;
  std::array<Tensor, kBatch> follower_normalized_values;
  std::array<Tensor, kBatch> follower_z;
  std::array<Tensor, kBatch> follower_activated_values;
  std::array<Tensor, kBatch> follower_gated_values;
  Tensor parent_recurrent;
  Tensor parent_normalized_heads;
  Tensor parent_normalized_values;
  Tensor parent_z;
  Tensor parent_activated_values;
  Tensor parent_gated_values;
  Tensor follower_flat_gated;
  Tensor parent_flat_gated;
  Tensor follower_output_tensor;
  Tensor parent_output_tensor;
};

double DecodeTailPercentile(std::vector<double> values, double quantile) {
  if (values.empty()) throw std::runtime_error("empty timing sample set");
  std::sort(values.begin(), values.end());
  const double position = quantile * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, values.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double DecodeTailMean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

template <typename Launch>
DecodeTailTiming TimeDecodeTail(std::string arm, DecodeTailFixture* fixture,
                                Launch launch) {
  for (std::uint64_t index = 0; index < kDecodeTailWarmups; ++index) {
    launch();
    fixture->Synchronize();
  }
  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;
  CheckAcl(aclrtCreateEvent(&start), "aclrtCreateEvent(start)");
  CheckAcl(aclrtCreateEvent(&end), "aclrtCreateEvent(end)");
  DecodeTailTiming timing{std::move(arm), {}, {}};
  try {
    for (std::uint64_t index = 0; index < kDecodeTailSamples; ++index) {
      const auto host_start = std::chrono::steady_clock::now();
      CheckAcl(aclrtRecordEvent(start, fixture->runtime.stream()),
               "aclrtRecordEvent(start)");
      launch();
      CheckAcl(aclrtRecordEvent(end, fixture->runtime.stream()),
               "aclrtRecordEvent(end)");
      fixture->Synchronize();
      const auto host_end = std::chrono::steady_clock::now();
      float elapsed_ms = 0.0F;
      CheckAcl(aclrtEventElapsedTime(&elapsed_ms, start, end),
               "aclrtEventElapsedTime");
      timing.device_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
      timing.host_us.push_back(
          std::chrono::duration<double, std::micro>(host_end - host_start)
              .count());
    }
  } catch (...) {
    (void)aclrtDestroyEvent(end);
    (void)aclrtDestroyEvent(start);
    throw;
  }
  CheckAcl(aclrtDestroyEvent(end), "aclrtDestroyEvent(end)");
  CheckAcl(aclrtDestroyEvent(start), "aclrtDestroyEvent(start)");
  return timing;
}

void EmitDecodeTailTiming(const DecodeTailTiming& timing, bool comma) {
  const auto emit = [](const std::vector<double>& values) {
    std::cout << "{\"mean\":" << DecodeTailMean(values)
              << ",\"p50\":" << DecodeTailPercentile(values, 0.5)
              << ",\"p95\":" << DecodeTailPercentile(values, 0.95)
              << ",\"min\":" << *std::min_element(values.begin(), values.end())
              << ",\"max\":" << *std::max_element(values.begin(), values.end())
              << '}';
  };
  std::cout << "{\"arm\":\"" << timing.arm << "\",\"device_us\":";
  emit(timing.device_us);
  std::cout << ",\"host_us\":";
  emit(timing.host_us);
  std::cout << '}' << (comma ? "," : "");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto device = argc > 1 ? std::stoi(argv[1]) : 4;
    if (device < 0 || device > 15) {
      throw std::runtime_error("device must be in [0,15]");
    }
    DecodeTailFixture fixture(device);
    fixture.LaunchV18FollowerTail();
    fixture.Synchronize();
    fixture.LaunchParentLocalTail();
    fixture.Synchronize();
    const auto output_elements = static_cast<std::size_t>(kBatch * kHidden);
    const auto comparison = Compare(
        "parent-local-tail",
        CopyBack(fixture.follower_output, output_elements),
        CopyBack(fixture.parent_output, output_elements));

    const auto c0 = TimeDecodeTail("C0-v18-follower-tail", &fixture, [&] {
      fixture.LaunchV18FollowerTail();
    });
    const auto h0 = TimeDecodeTail("H0-parent-local-tail", &fixture, [&] {
      fixture.LaunchParentLocalTail();
    });
    const auto h1 = TimeDecodeTail("H1-parent-local-tail", &fixture, [&] {
      fixture.LaunchParentLocalTail();
    });
    const auto c1 = TimeDecodeTail("C1-v18-follower-tail", &fixture, [&] {
      fixture.LaunchV18FollowerTail();
    });
    const auto p50 = [](const DecodeTailTiming& timing,
                        const std::vector<double> DecodeTailTiming::*member) {
      return DecodeTailPercentile(timing.*member, 0.5);
    };
    const double control_device =
        (p50(c0, &DecodeTailTiming::device_us) +
         p50(c1, &DecodeTailTiming::device_us)) /
        2.0;
    const double candidate_device =
        (p50(h0, &DecodeTailTiming::device_us) +
         p50(h1, &DecodeTailTiming::device_us)) /
        2.0;
    const double control_host =
        (p50(c0, &DecodeTailTiming::host_us) +
         p50(c1, &DecodeTailTiming::host_us)) /
        2.0;
    const double candidate_host =
        (p50(h0, &DecodeTailTiming::host_us) +
         p50(h1, &DecodeTailTiming::host_us)) /
        2.0;
    const bool exact = comparison.mismatched == 0 &&
                       comparison.max_abs == 0.0;

    std::cout << std::setprecision(12)
              << "{\"schema\":\"statecentric-qwen38-parent-local-decode-tail-"
                 "benchmark-v1\",\"device\":"
              << device << ",\"batch\":" << kBatch
              << ",\"tokens_per_sequence\":" << kDecodeTokens
              << ",\"warmups_per_arm\":" << kDecodeTailWarmups
              << ",\"samples_per_arm\":" << kDecodeTailSamples
              << ",\"removed_record_wait_edges\":2,\"arms\":[";
    EmitDecodeTailTiming(c0, true);
    EmitDecodeTailTiming(h0, true);
    EmitDecodeTailTiming(h1, true);
    EmitDecodeTailTiming(c1, false);
    std::cout << "],\"correctness\":{\"output_elements\":"
              << comparison.elements << ",\"mismatched\":"
              << comparison.mismatched << ",\"max_abs\":"
              << comparison.max_abs << ",\"exact\":"
              << (exact ? "true" : "false")
              << "},\"comparison\":{\"control_device_p50_us\":"
              << control_device << ",\"candidate_device_p50_us\":"
              << candidate_device << ",\"device_improvement\":"
              << control_device / candidate_device - 1.0
              << ",\"control_host_p50_us\":" << control_host
              << ",\"candidate_host_p50_us\":" << candidate_host
              << ",\"host_improvement\":"
              << control_host / candidate_host - 1.0
              << "},\"exact\":" << (exact ? "true" : "false")
              << ",\"component_only\":true,"
                 "\"online_performance_claim_permitted\":false}\n";
    return exact ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "Qwen3.8 parent-local decode tail benchmark failed: "
              << error.what() << '\n';
    return 2;
  }
}
