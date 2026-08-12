// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/openvino.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

constexpr size_t kWarmupIterations = 5;
constexpr size_t kTimedIterations = 30;

struct Profile {
    const char* metric_prefix;
    size_t batch;
    size_t sequence;
    size_t hidden;
    size_t input;
};

struct CompiledWorkload {
    ov::CompiledModel model;
    ov::InferRequest request;
    double compile_us;
    size_t lstm_primitive_count;
};

struct ProfileResult {
    double native_median_us;
    double decomposed_median_us;
    double native_compile_us;
    double decomposed_compile_us;
    size_t native_primitive_count;
    bool native_matches_decomposed;
};

ov::Tensor deterministic_tensor(const ov::element::Type& type, const ov::Shape& shape, size_t seed) {
    ov::Tensor tensor(type, shape);
    auto* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        const auto value = static_cast<float>(static_cast<int>((i + seed) % 29) - 14) / 64.0f;
        data[i] = ov::float16(value);
    }
    return tensor;
}

std::shared_ptr<ov::op::v0::Constant> deterministic_constant(const ov::Shape& shape, size_t seed) {
    return std::make_shared<ov::op::v0::Constant>(deterministic_tensor(ov::element::f16, shape, seed));
}

std::shared_ptr<ov::Model> make_native_model(const Profile& profile) {
    const ov::Shape x_shape{profile.batch, profile.sequence, profile.input};
    const ov::Shape state_shape{profile.batch, 2, profile.hidden};

    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, x_shape);
    auto h = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, state_shape);
    auto c = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, state_shape);
    auto sequence_lengths =
        ov::op::v0::Constant::create(ov::element::i64,
                                    {profile.batch},
                                    std::vector<int64_t>(profile.batch, static_cast<int64_t>(profile.sequence)));
    auto w = deterministic_constant({2, 4 * profile.hidden, profile.input}, 101);
    auto r = deterministic_constant({2, 4 * profile.hidden, profile.hidden}, 211);
    auto b = deterministic_constant({2, 4 * profile.hidden}, 307);

    auto lstm = std::make_shared<ov::op::v5::LSTMSequence>(
        x,
        h,
        c,
        sequence_lengths,
        w,
        r,
        b,
        profile.hidden,
        ov::op::RecurrentSequenceDirection::BIDIRECTIONAL,
        std::vector<float>{},
        std::vector<float>{},
        std::vector<std::string>{"sigmoid", "tanh", "tanh"},
        0.0f);
    lstm->set_friendly_name("artemis_native_bidirectional_lstm");

    return std::make_shared<ov::Model>(lstm->outputs(),
                                       ov::ParameterVector{x, h, c},
                                       std::string("artemis_native_") + profile.metric_prefix);
}

std::shared_ptr<ov::Model> make_decomposed_model(const Profile& profile) {
    const ov::Shape x_shape{profile.batch, profile.sequence, profile.input};
    const ov::Shape state_shape{profile.batch, 1, profile.hidden};

    // Separate parameters and constants make this an explicit two-branch
    // workload and prevent the plugin's sequence-fusion pass from rebuilding
    // a bidirectional LSTM before compilation.
    auto x_forward = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, x_shape);
    auto h_forward = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, state_shape);
    auto c_forward = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, state_shape);
    auto x_reverse = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, x_shape);
    auto h_reverse = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, state_shape);
    auto c_reverse = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, state_shape);

    auto make_direction = [&](const ov::Output<ov::Node>& x,
                              const ov::Output<ov::Node>& h,
                              const ov::Output<ov::Node>& c,
                              ov::op::RecurrentSequenceDirection direction) {
        const bool is_reverse = direction == ov::op::RecurrentSequenceDirection::REVERSE;
        const size_t w_offset = is_reverse ? 4 * profile.hidden * profile.input : 0;
        const size_t r_offset = is_reverse ? 4 * profile.hidden * profile.hidden : 0;
        const size_t b_offset = is_reverse ? 4 * profile.hidden : 0;
        auto sequence_lengths =
            ov::op::v0::Constant::create(ov::element::i64,
                                        {profile.batch},
                                        std::vector<int64_t>(profile.batch,
                                                             static_cast<int64_t>(profile.sequence)));
        return std::make_shared<ov::op::v5::LSTMSequence>(
            x,
            h,
            c,
            sequence_lengths,
            deterministic_constant({1, 4 * profile.hidden, profile.input}, 101 + w_offset),
            deterministic_constant({1, 4 * profile.hidden, profile.hidden}, 211 + r_offset),
            deterministic_constant({1, 4 * profile.hidden}, 307 + b_offset),
            profile.hidden,
            direction,
            std::vector<float>{},
            std::vector<float>{},
            std::vector<std::string>{"sigmoid", "tanh", "tanh"},
            0.0f);
    };

    auto forward = make_direction(x_forward,
                                  h_forward,
                                  c_forward,
                                  ov::op::RecurrentSequenceDirection::FORWARD);
    auto reverse = make_direction(x_reverse,
                                  h_reverse,
                                  c_reverse,
                                  ov::op::RecurrentSequenceDirection::REVERSE);
    forward->set_friendly_name("artemis_explicit_forward_lstm");
    reverse->set_friendly_name("artemis_explicit_reverse_lstm");

    ov::OutputVector outputs;
    outputs.reserve(3);
    for (size_t output = 0; output < 3; ++output) {
        outputs.push_back(std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{forward->output(output), reverse->output(output)},
            1));
    }

    return std::make_shared<ov::Model>(
        outputs,
        ov::ParameterVector{x_forward, h_forward, c_forward, x_reverse, h_reverse, c_reverse},
        std::string("artemis_decomposed_") + profile.metric_prefix);
}

CompiledWorkload compile_workload(ov::Core& core,
                                  const std::shared_ptr<ov::Model>& model,
                                  const std::vector<ov::Tensor>& input_tensors) {
    const auto compile_start = std::chrono::steady_clock::now();
    // The wrapper selects the oneDNN/OCL path through debug environment
    // settings before the plugin is loaded; these are not public properties.
    const ov::AnyMap config{
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
    };
    auto compiled_model = core.compile_model(model, "GPU", config);
    const auto compile_end = std::chrono::steady_clock::now();
    const auto compile_us =
        std::chrono::duration<double, std::micro>(compile_end - compile_start).count();

    // Implementation-name strings are not a stable contract across backends
    // (oneDNN's GPU RNN reports e.g. "ocl:simple:any"), so they are logged for
    // diagnosis while the metrics rely on the primitive count and the
    // native-vs-decomposed numerical comparison.
    size_t lstm_primitive_count = 0;
    std::cout << "[ARTEMIS_DIAG] " << model->get_friendly_name() << " LSTM_Seq impls:";
    for (const auto& node : compiled_model.get_runtime_model()->get_ordered_ops()) {
        const auto& info = node->get_rt_info();
        const auto layer_it = info.find(ov::exec_model_info::LAYER_TYPE);
        if (layer_it == info.end() || layer_it->second.as<std::string>() != "LSTM_Seq") {
            continue;
        }

        ++lstm_primitive_count;
        const auto impl_it = info.find(ov::exec_model_info::IMPL_TYPE);
        std::cout << ' ' << (impl_it != info.end() ? impl_it->second.as<std::string>() : "<unknown>");
    }
    std::cout << " (count=" << lstm_primitive_count << ')' << std::endl;

    auto request = compiled_model.create_infer_request();
    OPENVINO_ASSERT(input_tensors.size() == compiled_model.inputs().size());
    for (size_t input = 0; input < compiled_model.inputs().size(); ++input) {
        request.set_input_tensor(input, input_tensors[input]);
    }

    return {std::move(compiled_model),
            std::move(request),
            compile_us,
            lstm_primitive_count};
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2.0;
}

// Extract one direction slice [N,1,H] from a state tensor [N,2,H] so the
// decomposed workload consumes exactly the same problem as the native one.
ov::Tensor direction_slice(const ov::Tensor& state, size_t direction) {
    const auto& shape = state.get_shape();
    OPENVINO_ASSERT(shape.size() == 3 && shape[1] == 2);
    ov::Tensor slice(state.get_element_type(), ov::Shape{shape[0], 1, shape[2]});
    const auto* src = state.data<ov::float16>();
    auto* dst = slice.data<ov::float16>();
    for (size_t n = 0; n < shape[0]; ++n)
        for (size_t h = 0; h < shape[2]; ++h)
            dst[n * shape[2] + h] = src[(n * 2 + direction) * shape[2] + h];
    return slice;
}

bool tensors_match(const ov::Tensor& lhs, const ov::Tensor& rhs) {
    if (lhs.get_shape() != rhs.get_shape())
        return false;
    const auto* a = lhs.data<ov::float16>();
    const auto* b = rhs.data<ov::float16>();
    for (size_t i = 0; i < lhs.get_size(); ++i) {
        const float fa = static_cast<float>(a[i]);
        const float fb = static_cast<float>(b[i]);
        if (std::abs(fa - fb) > 5e-3f + 0.03f * std::abs(fb))
            return false;
    }
    return true;
}

double timed_infer(ov::InferRequest& request) {
    const auto start = std::chrono::steady_clock::now();
    request.infer();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - start).count();
}

ProfileResult run_profile(ov::Core& core, const Profile& profile) {
    // Both workloads consume the identical problem: the decomposed leg's
    // per-direction states are slices of the native leg's state tensors, so
    // the outputs are directly comparable at every batch size.
    const auto x = deterministic_tensor(ov::element::f16, {profile.batch, profile.sequence, profile.input}, 401);
    const auto h = deterministic_tensor(ov::element::f16, {profile.batch, 2, profile.hidden}, 438);
    const auto c = deterministic_tensor(ov::element::f16, {profile.batch, 2, profile.hidden}, 475);

    auto native = compile_workload(core, make_native_model(profile), {x, h, c});
    auto decomposed = compile_workload(core,
                                       make_decomposed_model(profile),
                                       {x,
                                        direction_slice(h, 0),
                                        direction_slice(c, 0),
                                        x,
                                        direction_slice(h, 1),
                                        direction_slice(c, 1)});

    for (size_t i = 0; i < kWarmupIterations; ++i) {
        native.request.infer();
        decomposed.request.infer();
    }

    // The decomposed graph is the numerically trusted reference: a native
    // candidate must never score speed on wrong results.
    bool matches = true;
    for (size_t output = 0; output < 3; ++output) {
        if (!tensors_match(native.request.get_output_tensor(output),
                           decomposed.request.get_output_tensor(output))) {
            matches = false;
            std::cout << "[ARTEMIS_DIAG] " << profile.metric_prefix
                      << " output " << output << " diverges from the decomposed reference" << std::endl;
        }
    }

    std::vector<double> native_times;
    std::vector<double> decomposed_times;
    native_times.reserve(kTimedIterations);
    decomposed_times.reserve(kTimedIterations);
    for (size_t i = 0; i < kTimedIterations; ++i) {
        if (i % 2 == 0) {
            native_times.push_back(timed_infer(native.request));
            decomposed_times.push_back(timed_infer(decomposed.request));
        } else {
            decomposed_times.push_back(timed_infer(decomposed.request));
            native_times.push_back(timed_infer(native.request));
        }
    }

    return {median(std::move(native_times)),
            median(std::move(decomposed_times)),
            native.compile_us,
            decomposed.compile_us,
            native.lstm_primitive_count,
            matches};
}

TEST(ArtemisLSTMBenchmark, DISABLED_NativeVsExplicitBidirectional) {
    ov::Core core;
    const Profile batch1{"bidir_b1", 1, 2, 128, 64};
    const Profile batch10{"bidir_b10", 10, 20, 10, 10};

    const auto b1 = run_profile(core, batch1);
    const auto b10 = run_profile(core, batch10);

    ASSERT_GT(b1.native_median_us, 0.0);
    ASSERT_GT(b1.decomposed_median_us, 0.0);
    ASSERT_GT(b10.native_median_us, 0.0);
    ASSERT_GT(b10.decomposed_median_us, 0.0);

    std::cout << std::setprecision(17)
              << "ARTEMIS_RESULTS_JSON={"
              << "\"bidir_b10_speedup_x\":" << b10.decomposed_median_us / b10.native_median_us << ","
              << "\"bidir_b1_speedup_x\":" << b1.decomposed_median_us / b1.native_median_us << ","
              << "\"bidir_b10_native_median_us\":" << b10.native_median_us << ","
              << "\"bidir_b10_decomposed_median_us\":" << b10.decomposed_median_us << ","
              << "\"bidir_b1_native_median_us\":" << b1.native_median_us << ","
              << "\"bidir_b1_decomposed_median_us\":" << b1.decomposed_median_us << ","
              << "\"bidir_b10_native_compile_us\":" << b10.native_compile_us << ","
              << "\"bidir_b10_decomposed_compile_us\":" << b10.decomposed_compile_us << ","
              << "\"bidir_b1_native_compile_us\":" << b1.native_compile_us << ","
              << "\"bidir_b1_decomposed_compile_us\":" << b1.decomposed_compile_us << ","
              << "\"bidir_b10_native_primitive_count\":" << b10.native_primitive_count << ","
              << "\"bidir_b1_native_primitive_count\":" << b1.native_primitive_count << ","
              << "\"bidir_b10_native_single_primitive\":" << (b10.native_primitive_count == 1 ? 1 : 0) << ","
              << "\"bidir_b1_native_single_primitive\":" << (b1.native_primitive_count == 1 ? 1 : 0) << ","
              << "\"bidir_b10_native_matches_decomposed\":" << (b10.native_matches_decomposed ? 1 : 0) << ","
              << "\"bidir_b1_native_matches_decomposed\":" << (b1.native_matches_decomposed ? 1 : 0)
              << "}" << std::endl;
}

}  // namespace
