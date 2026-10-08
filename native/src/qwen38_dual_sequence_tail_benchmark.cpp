#include <array>

#define main statecentric_qwen38_dual_sequence_tail_equivalence_main
#include "qwen38_dual_sequence_tail_equivalence_probe.cpp"
#undef main

#include <chrono>
#include <iomanip>
#include <numeric>

namespace {

constexpr std::uint64_t kBenchmarkWarmups = 3;
constexpr std::uint64_t kBenchmarkSamples = 12;

struct TailTiming {
  std::string arm;
  std::vector<double> device_us;
  std::vector<double> host_us;
};

struct TailBenchmarkFixture {
  explicit TailBenchmarkFixture(std::int32_t device) : runtime(device) {
    const auto recurrent_elements = static_cast<std::size_t>(
        kBatch * kTokens * kValueHeads * kHeadDim);
    const auto value_elements =
        static_cast<std::size_t>(kBatch * kTokens * kValue);
    const auto output_elements =
        static_cast<std::size_t>(kBatch * kTokens * kHidden);
    const auto weight_elements = static_cast<std::size_t>(kValue * kHidden);
    const auto recurrent_host = Generate(recurrent_elements, 0x13579BDFU);
    const auto z_host = Generate(value_elements, 0x2468ACE0U);
    const auto gamma_host = Generate(kHeadDim, 0xA5A5A5A5U);
    const auto weight_host = Generate(weight_elements, 0x5A5A5A5AU);
    recurrent = runtime.ToDevice(
        recurrent_host.data(), recurrent_host.size() * sizeof(std::uint16_t));
    z = runtime.ToDevice(z_host.data(),
                         z_host.size() * sizeof(std::uint16_t));
    gamma = runtime.ToDevice(
        gamma_host.data(), gamma_host.size() * sizeof(std::uint16_t));
    weight = runtime.ToDevice(
        weight_host.data(), weight_host.size() * sizeof(std::uint16_t));
    sequential_normalized =
        runtime.Allocate(value_elements * sizeof(std::uint16_t));
    sequential_activated =
        runtime.Allocate(value_elements * sizeof(std::uint16_t));
    sequential_gated = runtime.Allocate(value_elements * sizeof(std::uint16_t));
    sequential_output =
        runtime.Allocate(output_elements * sizeof(std::uint16_t));
    batched_normalized =
        runtime.Allocate(value_elements * sizeof(std::uint16_t));
    batched_activated =
        runtime.Allocate(value_elements * sizeof(std::uint16_t));
    batched_gated = runtime.Allocate(value_elements * sizeof(std::uint16_t));
    batched_output = runtime.Allocate(output_elements * sizeof(std::uint16_t));
    flat_output = runtime.Allocate(output_elements * sizeof(std::uint16_t));

    gamma_tensor = MakeContiguousTensor({kHeadDim}, ACL_BF16, gamma);
    sequential_weight = MakeTensor({kValue, kHidden}, {1, kValue},
                                   {kHidden, kValue}, ACL_BF16, weight);
    const auto value_sequence_bytes = static_cast<std::size_t>(
        kTokens * kValue * sizeof(std::uint16_t));
    const auto output_sequence_bytes = static_cast<std::size_t>(
        kTokens * kHidden * sizeof(std::uint16_t));
    for (std::int64_t batch = 0; batch < kBatch; ++batch) {
      const auto value_offset =
          static_cast<std::size_t>(batch) * value_sequence_bytes;
      const auto output_offset =
          static_cast<std::size_t>(batch) * output_sequence_bytes;
      sequential_recurrent[batch] = MakeContiguousTensor(
          {kTokens * kValueHeads, kHeadDim}, ACL_BF16,
          static_cast<std::uint8_t*>(recurrent) + value_offset);
      sequential_normalized_heads[batch] = MakeContiguousTensor(
          {kTokens * kValueHeads, kHeadDim}, ACL_BF16,
          static_cast<std::uint8_t*>(sequential_normalized) + value_offset);
      sequential_normalized_values[batch] = MakeContiguousTensor(
          {kTokens, kValue}, ACL_BF16,
          static_cast<std::uint8_t*>(sequential_normalized) + value_offset);
      sequential_z[batch] = MakeContiguousTensor(
          {kTokens, kValue}, ACL_BF16,
          static_cast<std::uint8_t*>(z) + value_offset);
      sequential_activated_values[batch] = MakeContiguousTensor(
          {kTokens, kValue}, ACL_BF16,
          static_cast<std::uint8_t*>(sequential_activated) + value_offset);
      sequential_gated_values[batch] = MakeContiguousTensor(
          {kTokens, kValue}, ACL_BF16,
          static_cast<std::uint8_t*>(sequential_gated) + value_offset);
      sequential_outputs[batch] = MakeContiguousTensor(
          {kTokens, kHidden}, ACL_BF16,
          static_cast<std::uint8_t*>(sequential_output) + output_offset);
    }
    batched_recurrent_tensor = MakeContiguousTensor(
        {kBatch, kTokens * kValueHeads, kHeadDim}, ACL_BF16, recurrent);
    batched_normalized_heads = MakeContiguousTensor(
        {kBatch, kTokens * kValueHeads, kHeadDim}, ACL_BF16,
        batched_normalized);
    batched_normalized_values = MakeContiguousTensor(
        {kBatch, kTokens, kValue}, ACL_BF16, batched_normalized);
    batched_z_tensor = MakeContiguousTensor(
        {kBatch, kTokens, kValue}, ACL_BF16, z);
    batched_activated_values = MakeContiguousTensor(
        {kBatch, kTokens, kValue}, ACL_BF16, batched_activated);
    batched_gated_values = MakeContiguousTensor(
        {kBatch, kTokens, kValue}, ACL_BF16, batched_gated);
    batched_weight = MakeTensor(
        {1, kValue, kHidden}, {kValue * kHidden, 1, kValue},
        {1, kHidden, kValue}, ACL_BF16, weight);
    batched_output_tensor = MakeContiguousTensor(
        {kBatch, kTokens, kHidden}, ACL_BF16, batched_output);
    flat_gated_values = MakeContiguousTensor(
        {kBatch * kTokens, kValue}, ACL_BF16, batched_gated);
    flat_output_tensor = MakeContiguousTensor(
        {kBatch * kTokens, kHidden}, ACL_BF16, flat_output);
  }

  void LaunchSequential() {
    for (std::int64_t batch = 0; batch < kBatch; ++batch) {
      EnqueueRmsNorm(runtime, sequential_recurrent[batch], gamma_tensor,
                     sequential_normalized_heads[batch],
                     {kTokens * kValueHeads, 1});
      EnqueueSilu(runtime, sequential_z[batch],
                  sequential_activated_values[batch]);
      EnqueueMul(runtime, sequential_activated_values[batch],
                 sequential_normalized_values[batch],
                 sequential_gated_values[batch]);
      EnqueueMatmul(runtime, sequential_gated_values[batch],
                    sequential_weight, sequential_outputs[batch]);
    }
  }

  void LaunchBatched() {
    EnqueueRmsNorm(runtime, batched_recurrent_tensor, gamma_tensor,
                   batched_normalized_heads,
                   {kBatch, kTokens * kValueHeads, 1});
    EnqueueSilu(runtime, batched_z_tensor, batched_activated_values);
    EnqueueMul(runtime, batched_activated_values, batched_normalized_values,
               batched_gated_values);
    EnqueueBatchMatmul(runtime, batched_gated_values, batched_weight,
                       batched_output_tensor);
  }

  void LaunchFlatBatched() {
    EnqueueRmsNorm(runtime, batched_recurrent_tensor, gamma_tensor,
                   batched_normalized_heads,
                   {kBatch, kTokens * kValueHeads, 1});
    EnqueueSilu(runtime, batched_z_tensor, batched_activated_values);
    EnqueueMul(runtime, batched_activated_values, batched_normalized_values,
               batched_gated_values);
    EnqueueMatmul(runtime, flat_gated_values, sequential_weight,
                  flat_output_tensor);
  }

  Runtime runtime;
  void* recurrent = nullptr;
  void* z = nullptr;
  void* gamma = nullptr;
  void* weight = nullptr;
  void* sequential_normalized = nullptr;
  void* sequential_activated = nullptr;
  void* sequential_gated = nullptr;
  void* sequential_output = nullptr;
  void* batched_normalized = nullptr;
  void* batched_activated = nullptr;
  void* batched_gated = nullptr;
  void* batched_output = nullptr;
  void* flat_output = nullptr;
  Tensor gamma_tensor;
  Tensor sequential_weight;
  std::array<Tensor, kBatch> sequential_recurrent;
  std::array<Tensor, kBatch> sequential_normalized_heads;
  std::array<Tensor, kBatch> sequential_normalized_values;
  std::array<Tensor, kBatch> sequential_z;
  std::array<Tensor, kBatch> sequential_activated_values;
  std::array<Tensor, kBatch> sequential_gated_values;
  std::array<Tensor, kBatch> sequential_outputs;
  Tensor batched_recurrent_tensor;
  Tensor batched_normalized_heads;
  Tensor batched_normalized_values;
  Tensor batched_z_tensor;
  Tensor batched_activated_values;
  Tensor batched_gated_values;
  Tensor batched_weight;
  Tensor batched_output_tensor;
  Tensor flat_gated_values;
  Tensor flat_output_tensor;
};

double Percentile(std::vector<double> values, double quantile) {
  if (values.empty()) throw std::runtime_error("empty timing sample set");
  std::sort(values.begin(), values.end());
  const double position = quantile * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, values.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

template <typename Launch>
TailTiming RunTiming(std::string arm, TailBenchmarkFixture* fixture,
                     Launch launch) {
  for (std::uint64_t index = 0; index < kBenchmarkWarmups; ++index) {
    launch();
    fixture->runtime.Synchronize();
  }
  aclrtEvent start = nullptr;
  aclrtEvent end = nullptr;
  CheckAcl(aclrtCreateEvent(&start), "aclrtCreateEvent(start)");
  CheckAcl(aclrtCreateEvent(&end), "aclrtCreateEvent(end)");
  TailTiming result{std::move(arm), {}, {}};
  result.device_us.reserve(kBenchmarkSamples);
  result.host_us.reserve(kBenchmarkSamples);
  try {
    for (std::uint64_t index = 0; index < kBenchmarkSamples; ++index) {
      const auto host_start = std::chrono::steady_clock::now();
      CheckAcl(aclrtRecordEvent(start, fixture->runtime.stream()),
               "aclrtRecordEvent(start)");
      launch();
      CheckAcl(aclrtRecordEvent(end, fixture->runtime.stream()),
               "aclrtRecordEvent(end)");
      fixture->runtime.Synchronize();
      const auto host_end = std::chrono::steady_clock::now();
      float elapsed_ms = 0.0F;
      CheckAcl(aclrtEventElapsedTime(&elapsed_ms, start, end),
               "aclrtEventElapsedTime");
      result.device_us.push_back(static_cast<double>(elapsed_ms) * 1000.0);
      result.host_us.push_back(
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
  return result;
}

double Mean(const std::vector<double>& values) {
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

void EmitTiming(const TailTiming& timing, bool comma) {
  const auto emit = [](const std::vector<double>& values) {
    std::cout << "{\"mean\":" << Mean(values)
              << ",\"p50\":" << Percentile(values, 0.5)
              << ",\"p95\":" << Percentile(values, 0.95)
              << ",\"min\":"
              << *std::min_element(values.begin(), values.end())
              << ",\"max\":"
              << *std::max_element(values.begin(), values.end()) << '}';
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
    TailBenchmarkFixture fixture(device);
    const auto c0 = RunTiming("C0-sequential", &fixture,
                              [&] { fixture.LaunchSequential(); });
    const auto h0 = RunTiming("H0-batched", &fixture,
                              [&] { fixture.LaunchBatched(); });
    const auto f0 = RunTiming("F0-flat-batched", &fixture,
                              [&] { fixture.LaunchFlatBatched(); });
    const auto f1 = RunTiming("F1-flat-batched", &fixture,
                              [&] { fixture.LaunchFlatBatched(); });
    const auto h1 = RunTiming("H1-batched", &fixture,
                              [&] { fixture.LaunchBatched(); });
    const auto c1 = RunTiming("C1-sequential", &fixture,
                              [&] { fixture.LaunchSequential(); });
    const auto output_elements =
        static_cast<std::size_t>(kBatch * kTokens * kHidden);
    const auto comparison = Compare(
        "output", CopyBack(fixture.sequential_output, output_elements),
        CopyBack(fixture.batched_output, output_elements));
    const auto flat_comparison = Compare(
        "flat-output", CopyBack(fixture.sequential_output, output_elements),
        CopyBack(fixture.flat_output, output_elements));
    const double control_device =
        (Percentile(c0.device_us, 0.5) + Percentile(c1.device_us, 0.5)) / 2.0;
    const double candidate_device =
        (Percentile(h0.device_us, 0.5) + Percentile(h1.device_us, 0.5)) / 2.0;
    const double flat_device =
        (Percentile(f0.device_us, 0.5) + Percentile(f1.device_us, 0.5)) / 2.0;
    const double control_host =
        (Percentile(c0.host_us, 0.5) + Percentile(c1.host_us, 0.5)) / 2.0;
    const double candidate_host =
        (Percentile(h0.host_us, 0.5) + Percentile(h1.host_us, 0.5)) / 2.0;
    const double flat_host =
        (Percentile(f0.host_us, 0.5) + Percentile(f1.host_us, 0.5)) / 2.0;
    const bool device_direction_consistent =
        Percentile(h0.device_us, 0.5) < control_device &&
        Percentile(h1.device_us, 0.5) < control_device;
    const bool host_no_material_regression = candidate_host <= control_host * 1.01;
    const bool supported = comparison.mismatched == 0 &&
                           flat_comparison.mismatched == 0 &&
                           device_direction_consistent &&
                           host_no_material_regression;
    std::cout << std::setprecision(12)
              << "{\"schema\":\"statecentric-qwen38-dual-sequence-tail-"
                 "benchmark-v2\",\"device\":"
              << device << ",\"batch\":" << kBatch
              << ",\"tokens_per_sequence\":" << kTokens
              << ",\"warmups_per_arm\":" << kBenchmarkWarmups
              << ",\"samples_per_arm\":" << kBenchmarkSamples
              << ",\"order\":[\"C0-sequential\",\"H0-batched\","
                 "\"F0-flat-batched\",\"F1-flat-batched\","
                 "\"H1-batched\",\"C1-sequential\"],\"arms\":[";
    EmitTiming(c0, true);
    EmitTiming(h0, true);
    EmitTiming(f0, true);
    EmitTiming(f1, true);
    EmitTiming(h1, true);
    EmitTiming(c1, false);
    std::cout << "],\"correctness\":{\"output_elements\":"
              << comparison.elements << ",\"mismatched\":"
              << comparison.mismatched << ",\"max_abs\":"
              << comparison.max_abs << ",\"exact\":"
              << (comparison.mismatched == 0 && comparison.max_abs == 0.0
                      ? "true"
                      : "false")
              << "},\"flat_correctness\":{\"output_elements\":"
              << flat_comparison.elements << ",\"mismatched\":"
              << flat_comparison.mismatched << ",\"max_abs\":"
              << flat_comparison.max_abs << ",\"exact\":"
              << (flat_comparison.mismatched == 0 &&
                          flat_comparison.max_abs == 0.0
                      ? "true"
                      : "false")
              << "},\"comparison\":{\"control_device_p50_us\":"
              << control_device << ",\"candidate_device_p50_us\":"
              << candidate_device << ",\"device_improvement\":"
              << control_device / candidate_device - 1.0
              << ",\"flat_device_p50_us\":" << flat_device
              << ",\"flat_device_improvement\":"
              << control_device / flat_device - 1.0
              << ",\"flat_vs_batch_device_improvement\":"
              << candidate_device / flat_device - 1.0
              << ",\"control_host_p50_us\":" << control_host
              << ",\"candidate_host_p50_us\":" << candidate_host
              << ",\"host_improvement\":"
              << control_host / candidate_host - 1.0
              << ",\"flat_host_p50_us\":" << flat_host
              << ",\"flat_host_improvement\":"
              << control_host / flat_host - 1.0
              << ",\"flat_vs_batch_host_improvement\":"
              << candidate_host / flat_host - 1.0
              << ",\"device_direction_consistent\":"
              << (device_direction_consistent ? "true" : "false")
              << ",\"host_no_material_regression\":"
              << (host_no_material_regression ? "true" : "false")
              << ",\"integration_probe_supported\":"
              << (supported ? "true" : "false")
              << "},\"component_only\":true,"
                 "\"online_performance_claim_permitted\":false}\n";
    return comparison.mismatched == 0 && comparison.max_abs == 0.0 &&
                   flat_comparison.mismatched == 0 &&
                   flat_comparison.max_abs == 0.0
               ? 0
               : 1;
  } catch (const std::exception& error) {
    std::cerr << "Qwen3.8 dual-sequence tail benchmark failed: "
              << error.what() << '\n';
    return 2;
  }
}
