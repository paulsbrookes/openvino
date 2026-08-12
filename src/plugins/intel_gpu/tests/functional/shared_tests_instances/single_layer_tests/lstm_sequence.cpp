// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include "single_op_tests/lstm_sequence.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/properties.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"

namespace {
using ov::test::LSTMSequenceTest;
using ov::test::utils::SequenceTestsMode;
using ov::test::utils::InputLayerType;

std::vector<std::string> get_lstm_implementations(const ov::CompiledModel& model) {
    std::vector<std::string> implementations;
    for (const auto& node : model.get_runtime_model()->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto layer_type = rt_info.at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
        if (layer_type != "LSTM_Seq")
            continue;

        implementations.push_back(rt_info.at(ov::exec_model_info::IMPL_TYPE).as<std::string>());
    }
    return implementations;
}

void log_lstm_implementations(const ov::CompiledModel& model, const char* tag) {
    const auto implementations = get_lstm_implementations(model);
    std::cout << "[ARTEMIS_DIAG] " << tag << " LSTM_Seq count=" << implementations.size();
    for (const auto& impl_type : implementations)
        std::cout << " impl=" << impl_type;
    std::cout << std::endl;
}

// Backend implementation-name strings are not a stable contract (for example,
// oneDNN's GPU RNN reports "ocl:simple:any"), so the topology gate asserts only
// the primitive count. Implementation names are logged for diagnosis; numerical
// gates decide correctness.
void check_bidirectional_lstm_topology(const ov::CompiledModel& model,
                                       size_t expected_native_count,
                                       size_t expected_decomposed_count) {
    const auto implementations = get_lstm_implementations(model);
    EXPECT_TRUE(implementations.size() == expected_native_count ||
                implementations.size() == expected_decomposed_count)
        << "Expected the native topology (" << expected_native_count
        << " LSTM_Seq primitives) or the baseline decomposition ("
        << expected_decomposed_count << "), got " << implementations.size();
}

class LSTMSequenceGPUTest : public LSTMSequenceTest {
protected:
    void SetUp() override {
        LSTMSequenceTest::SetUp();
        ov::test::LSTMSequenceParams params;
        params = this->GetParam();
        const auto network_precision = std::get<9>(params);
        const auto activations = std::get<5>(params);
        if (network_precision == ov::element::f16) {
            rel_threshold = 0.03f;
            abs_threshold = 0.0025f;
            if (activations == std::vector<std::string>{"tanh", "tanh", "tanh"}) {
                rel_threshold = 0.05f;
                abs_threshold = 0.005f;
            }
        }
    }
};

class LSTMSequenceOneDNNGPUTest : public LSTMSequenceGPUTest {
protected:
    void SetUp() override {
        LSTMSequenceGPUTest::SetUp();

        for (const auto& node : function->get_ordered_ops()) {
            if (ov::is_type<ov::op::v5::LSTMSequence>(node))
                node->set_friendly_name("correctness_lstm");
        }

        if (std::get<9>(GetParam()) == ov::element::f32)
            configuration.insert(ov::hint::inference_precision(ov::element::f32));
    }

    void compare_with_ocl() {
        auto ocl_function = function->clone();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
        manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        manager.run_passes(ocl_function);

        ov::AnyMap ocl_config;

        const auto model_type = std::get<9>(GetParam());
        if (model_type == ov::element::f32)
            ocl_config.insert(ov::hint::inference_precision(ov::element::f32));

        auto ocl_model = core->compile_model(ocl_function, targetDevice, ocl_config);
        auto ocl_request = ocl_model.create_infer_request();
        const auto& parameters = function->get_parameters();
        for (size_t i = 0; i < parameters.size(); ++i)
            ocl_request.set_input_tensor(i, inputs.at(parameters[i]));
        ocl_request.infer();

        std::vector<ov::Tensor> onednn_outputs;
        std::vector<ov::Tensor> ocl_outputs;
        for (size_t i = 0; i < function->get_output_size(); ++i) {
            onednn_outputs.push_back(inferRequest.get_output_tensor(i));
            ocl_outputs.push_back(ocl_request.get_output_tensor(i));
        }

        compare(onednn_outputs, ocl_outputs);
    }
};

class LSTMSequenceFallbackGPUTest : public LSTMSequenceGPUTest {
protected:
    void SetUp() override {
        LSTMSequenceGPUTest::SetUp();
    }
};

std::vector<ov::test::utils::SequenceTestsMode> mode{ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_CONST,
                                                     ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST,
                                                     ov::test::utils::SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM,
                                                     ov::test::utils::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST,
                                                     ov::test::utils::SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM,
                                                     ov::test::utils::SequenceTestsMode::PURE_SEQ};
// output values increase rapidly without clip, so use only seq_lengths = 2
std::vector<size_t> seq_lengths_zero_clip{2};
std::vector<size_t> seq_lengths_clip_non_zero{20};
std::vector<size_t> batch{10};
std::vector<size_t> hidden_size{1, 4, 10};
std::vector<size_t> hidden_size_smoke{1};
std::vector<size_t> input_size{10};
std::vector<std::vector<std::string>> activations = {{"relu", "sigmoid", "tanh"}, {"sigmoid", "tanh", "tanh"},
                                                     {"tanh", "relu", "sigmoid"}, {"sigmoid", "sigmoid", "sigmoid"},
                                                     {"tanh", "tanh", "tanh"}, {"relu", "relu", "relu"}};
std::vector<std::vector<std::string>> activations_smoke = {{"relu", "sigmoid", "tanh"}};
std::vector<float> clip{0.f};
std::vector<float> clip_non_zeros{0.7f};
std::vector<ov::op::RecurrentSequenceDirection> direction = {ov::op::RecurrentSequenceDirection::FORWARD,
                                                             ov::op::RecurrentSequenceDirection::REVERSE,
                                                             ov::op::RecurrentSequenceDirection::BIDIRECTIONAL
};
std::vector<ov::element::Type> netPrecisions = {ov::element::f32,
                                                ov::element::f16};
TEST_P(LSTMSequenceGPUTest, Inference) {
    run();
};

// Numerics vs the framework reference and topology sanity are asserted in
// separate tests from decomposed-graph parity, so one failure names one cause.
TEST_P(LSTMSequenceOneDNNGPUTest, Numerics) {
    run();
    log_lstm_implementations(compiledModel, "Numerics");
    check_bidirectional_lstm_topology(compiledModel, 1, 2);
}

TEST_P(LSTMSequenceOneDNNGPUTest, ParityWithDecomposedReference) {
    run();
    log_lstm_implementations(compiledModel, "ParityWithDecomposedReference");
    compare_with_ocl();
}

TEST_P(LSTMSequenceFallbackGPUTest, UnsupportedClipFallsBack) {
    run();
    EXPECT_TRUE(get_lstm_implementations(compiledModel).empty());
}

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSequenceBidirectionalOneDNNCorrectness,
                         LSTMSequenceOneDNNGPUTest,
                         ::testing::Combine(
                             ::testing::Values(ov::test::utils::SequenceTestsMode::PURE_SEQ),
                             ::testing::Values(3),
                             ::testing::Values(1, 4),
                             ::testing::Values(4),
                             ::testing::Values(3),
                             ::testing::Values(std::vector<std::string>{"sigmoid", "tanh", "tanh"}),
                             ::testing::Values(0.f),
                             ::testing::Values(ov::op::RecurrentSequenceDirection::BIDIRECTIONAL),
                             ::testing::Values(ov::test::utils::InputLayerType::CONSTANT,
                                               ov::test::utils::InputLayerType::PARAMETER),
                             ::testing::Values(ov::element::f16, ov::element::f32),
                             ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         LSTMSequenceOneDNNGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSequenceOneDNNFallback,
                         LSTMSequenceFallbackGPUTest,
                         ::testing::Combine(
                             ::testing::Values(ov::test::utils::SequenceTestsMode::PURE_SEQ),
                             ::testing::Values(3),
                             ::testing::Values(1),
                             ::testing::Values(4),
                             ::testing::Values(3),
                             ::testing::Values(std::vector<std::string>{"sigmoid", "tanh", "tanh"}),
                             ::testing::Values(0.7f),
                             ::testing::Values(ov::op::RecurrentSequenceDirection::BIDIRECTIONAL),
                             ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                             ::testing::Values(ov::element::f16),
                             ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         LSTMSequenceFallbackGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(LSTMSequenceCommonZeroClip, LSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(mode),
                                ::testing::ValuesIn(seq_lengths_zero_clip),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(LSTMSequenceCommonZeroClipNonConstantWRB, LSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::SequenceTestsMode::PURE_SEQ),
                                ::testing::ValuesIn(seq_lengths_zero_clip),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(LSTMSequenceCommonClip, LSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(mode),
                                ::testing::ValuesIn(seq_lengths_clip_non_zero),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations),
                                ::testing::ValuesIn(clip_non_zeros),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_LSTMSequenceCommonClip, LSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(mode),
                                ::testing::ValuesIn(seq_lengths_clip_non_zero),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size_smoke),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(activations_smoke),
                                ::testing::ValuesIn(clip_non_zeros),
                                ::testing::ValuesIn(direction),
                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);


std::vector<size_t> seq_lengths_cm{2};
std::vector<size_t> batch_cm{1};
std::vector<size_t> hidden_size_cm{128};
std::vector<size_t> input_size_cm{64, 256};
std::vector<std::vector<std::string>> activations_cm = {{"sigmoid", "tanh", "tanh"}};
std::vector<float> clip_cm{0};
std::vector<ov::element::Type> netPrecisions_cm = {ov::element::f16};

INSTANTIATE_TEST_SUITE_P(LSTMSequenceCM, LSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::SequenceTestsMode::PURE_SEQ),
                                ::testing::ValuesIn(seq_lengths_cm),
                                ::testing::ValuesIn(batch_cm),
                                ::testing::ValuesIn(hidden_size_cm),
                                ::testing::ValuesIn(input_size_cm),
                                ::testing::ValuesIn(activations_cm),
                                ::testing::ValuesIn(clip_cm),
                                ::testing::Values(ov::op::RecurrentSequenceDirection::BIDIRECTIONAL),
                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                ::testing::ValuesIn(netPrecisions_cm),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);

class MultipleLSTMSequenceGPUTest : public LSTMSequenceTest {
private:
    size_t max_seq_lengths;

public:
    void SetUp() override {
    SequenceTestsMode mode;
    size_t seq_lengths;
    size_t batch;
    size_t hidden_size;
    size_t input_size;
    std::vector<std::string> activations;
    std::vector<float> activations_alpha;
    std::vector<float> activations_beta;
    float clip;
    ov::op::RecurrentSequenceDirection direction;
    InputLayerType WRBType;
    ov::element::Type model_type;
    std::tie(mode, seq_lengths, batch, hidden_size, input_size, activations, clip, direction,
                WRBType, model_type, targetDevice) = this->GetParam();

    max_seq_lengths = seq_lengths;
    size_t num_directions = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
    std::vector<ov::Shape> inputShapes = {
            {batch, seq_lengths, input_size},
            {batch, num_directions, hidden_size},
            {batch, num_directions, hidden_size},
            {batch},
            {num_directions, 4 * hidden_size, input_size},
            {num_directions, 4 * hidden_size, hidden_size},
            {num_directions, 4 * hidden_size},
    };

    const auto& W_shape = inputShapes[4];
    const auto& R_shape = inputShapes[5];
    const auto& B_shape = inputShapes[6];

    std::vector<ov::Shape> param_shapes{inputShapes[0], inputShapes[1], inputShapes[2]};
    std::vector<ov::Shape> const_input_shapes;
    if (mode == SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
        mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
        mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
        param_shapes.push_back(inputShapes[3]);
    }

    if (WRBType == InputLayerType::PARAMETER) {
        param_shapes.push_back(inputShapes[4]);
        param_shapes.push_back(inputShapes[5]);
        param_shapes.push_back(inputShapes[6]);
    }
    init_input_shapes(ov::test::static_shapes_to_test_representation(param_shapes));

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                               std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1]),
                               std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[2])};

    std::shared_ptr<ov::Node> seq_lengths_node;
    if (mode == SequenceTestsMode::CONVERT_TO_TI_MAX_SEQ_LEN_PARAM ||
        mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_PARAM ||
        mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM) {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[3]);
        seq_lengths_node = param;
        seq_lengths_node->set_friendly_name("seq_lengths");
        params.push_back(param);
    } else if (mode == SequenceTestsMode::CONVERT_TO_TI_RAND_SEQ_LEN_CONST ||
               mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST) {
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = seq_lengths;
        auto tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, inputShapes[3], in_data);
        seq_lengths_node = std::make_shared<ov::op::v0::Constant>(tensor);
    } else {
        std::vector<int64_t> lengths(inputShapes[3][0], seq_lengths);
        seq_lengths_node = ov::op::v0::Constant::create(ov::element::i64, inputShapes[3], lengths);
    }

    std::shared_ptr<ov::Node> W, R, B;
    if (WRBType == InputLayerType::PARAMETER) {
        auto param_num = inputDynamicShapes.size();
        const auto W_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[param_num - 3]);
        const auto R_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[param_num - 2]);
        const auto B_param = std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[param_num - 1]);
        W = W_param;
        R = R_param;
        B = B_param;
        params.push_back(W_param);
        params.push_back(R_param);
        params.push_back(B_param);
    } else {
        auto tensor_w = ov::test::utils::create_and_fill_tensor_real_distribution(model_type, W_shape, -1, 1, 0);
        W = std::make_shared<ov::op::v0::Constant>(tensor_w);

        auto tensor_r = ov::test::utils::create_and_fill_tensor_real_distribution(model_type, R_shape, -1, 1, 0);
        R = std::make_shared<ov::op::v0::Constant>(tensor_r);

        auto tensor_b = ov::test::utils::create_and_fill_tensor_real_distribution(model_type, B_shape, -1, 1, 0);
        B = std::make_shared<ov::op::v0::Constant>(tensor_b);
    }

    auto lstm_sequence0 = std::make_shared<ov::op::v5::LSTMSequence>(params[0], params[1], params[2], seq_lengths_node, W, R, B, hidden_size, direction,
                                                                    std::vector<float>{}, std::vector<float>{}, activations, clip);
    auto lstm_sequence1 = std::make_shared<ov::op::v5::LSTMSequence>(params[0], lstm_sequence0->output(1), lstm_sequence0->output(2),
                                                                    seq_lengths_node, W, R, B, hidden_size, direction,
                                                                    std::vector<float>{}, std::vector<float>{}, activations, clip);
    auto lstm_sequence2 = std::make_shared<ov::op::v5::LSTMSequence>(params[0], lstm_sequence0->output(1), lstm_sequence0->output(2),
                                                                    seq_lengths_node, W, R, B, hidden_size, direction,
                                                                    std::vector<float>{}, std::vector<float>{}, activations, clip);
    lstm_sequence0->set_friendly_name("shared_weights_lstm_0");
    lstm_sequence1->set_friendly_name("shared_weights_lstm_1");
    lstm_sequence2->set_friendly_name("shared_weights_lstm_2");
    auto concat0 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lstm_sequence1->output(0), lstm_sequence2->output(0)}, 0);
    auto concat1 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lstm_sequence1->output(1), lstm_sequence2->output(1)}, 0);
    auto concat2 = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{lstm_sequence1->output(2), lstm_sequence2->output(2)}, 0);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat0),
                             std::make_shared<ov::op::v0::Result>(concat1),
                             std::make_shared<ov::op::v0::Result>(concat2)};

    function = std::make_shared<ov::Model>(results, params, "multiple_lstm_sequence");
    if (model_type == ov::element::f32)
        configuration.insert(ov::hint::inference_precision(ov::element::f32));
    bool is_pure_sequence = mode == SequenceTestsMode::PURE_SEQ ||
                            mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_PARAM ||
                            mode == SequenceTestsMode::PURE_SEQ_RAND_SEQ_LEN_CONST;

    if (!is_pure_sequence) {
        ov::pass::Manager manager;
        if (direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL)
            manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
        manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        manager.run_passes(function);
        bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, true);
    } else {
        bool ti_found = ov::test::utils::is_tensor_iterator_exist(function);
        EXPECT_EQ(ti_found, false);
    }
    if (model_type == ov::element::f16) {
    rel_threshold = 0.03f;
    abs_threshold = 0.0025f;
        if (activations == std::vector<std::string>{"tanh", "tanh", "tanh"}) {
            rel_threshold = 0.05f;
            abs_threshold = 0.005f;
        }
    }
}

void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
    inputs.clear();
    const auto& func_inputs = function->inputs();
    ov::test::utils::InputGenerateData in_data;
    in_data.start_from = 0;
    in_data.range = 10;

    for (size_t i = 0; i < func_inputs.size(); ++i) {
        ov::Tensor tensor;

        if (i == 3) {
            in_data.range = max_seq_lengths;
        } else {
            in_data.range = 10;
        }
        if (i < 3) {
            tensor = ov::test::utils::create_and_fill_tensor_real_distribution(func_inputs[i].get_element_type(), targetInputStaticShapes[i], 0, 1, 0);
        } else {
            tensor = ov::test::utils::create_and_fill_tensor(func_inputs[i].get_element_type(), targetInputStaticShapes[i], in_data);
        }
        inputs.insert({func_inputs[i].get_node_shared_ptr(), tensor});
    }
}
};

std::vector<std::vector<std::string>> multilstmseq_activations = {{"sigmoid", "tanh", "tanh"}};

TEST_P(MultipleLSTMSequenceGPUTest, Inference) {
    run();
    const auto direction = std::get<7>(GetParam());
    if (direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        log_lstm_implementations(compiledModel, "MultipleLSTMSequence");
        check_bidirectional_lstm_topology(compiledModel, 2, 4);
    }
};

// Check whether inputs of LSTMSequence have right idx and weight optimization works properly when constant is shared by multiple LSTMSequence
// for onednn primitive
INSTANTIATE_TEST_SUITE_P(MultipleLSTMSequenceGPUTest, MultipleLSTMSequenceGPUTest,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::SequenceTestsMode::PURE_SEQ),
                                ::testing::ValuesIn(seq_lengths_zero_clip),
                                ::testing::ValuesIn(batch),
                                ::testing::ValuesIn(hidden_size),
                                ::testing::ValuesIn(input_size),
                                ::testing::ValuesIn(multilstmseq_activations),
                                ::testing::ValuesIn(clip),
                                ::testing::Values(ov::op::RecurrentSequenceDirection::FORWARD),
                                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                ::testing::Values(ov::element::f16),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        LSTMSequenceGPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MultipleBidirectionalLSTMSequenceSharedWeightsOneDNN,
                         MultipleLSTMSequenceGPUTest,
                         ::testing::Combine(
                             ::testing::Values(ov::test::utils::SequenceTestsMode::PURE_SEQ),
                             ::testing::Values(3),
                             ::testing::Values(1, 4),
                             ::testing::Values(4),
                             ::testing::Values(3),
                             ::testing::ValuesIn(multilstmseq_activations),
                             ::testing::ValuesIn(clip),
                             ::testing::Values(ov::op::RecurrentSequenceDirection::BIDIRECTIONAL),
                             ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                             ::testing::Values(ov::element::f16, ov::element::f32),
                             ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         LSTMSequenceGPUTest::getTestCaseName);

// Always-passing diagnostic: prints device identity and the exec-graph
// LAYER_TYPE/IMPL_TYPE reality for unidirectional and bidirectional
// LSTMSequence so every runner log carries the environment ground truth.
TEST(ArtemisLSTMDiagnostics, PrintEnvironment) {
    ov::Core core;
    try {
        std::cout << "[ARTEMIS_DIAG] device=" << core.get_property("GPU", ov::device::full_name) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "[ARTEMIS_DIAG] device=unavailable (" << e.what() << ")" << std::endl;
    }
    try {
        std::cout << "[ARTEMIS_DIAG] architecture=" << core.get_property("GPU", ov::device::architecture) << std::endl;
    } catch (const std::exception&) {
        std::cout << "[ARTEMIS_DIAG] architecture=unavailable" << std::endl;
    }

    const auto make_model = [](ov::op::RecurrentSequenceDirection direction) {
        constexpr size_t batch = 2, seq = 2, input = 3, hidden = 4;
        const size_t dirs = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? 2 : 1;
        const auto type = ov::element::f16;
        auto x = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{batch, seq, input});
        auto h = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{batch, dirs, hidden});
        auto c = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape{batch, dirs, hidden});
        auto seq_lengths = ov::op::v0::Constant::create(ov::element::i64, {batch},
                                                        std::vector<int64_t>(batch, static_cast<int64_t>(seq)));
        auto weight = [&](const ov::Shape& shape, float value) {
            return ov::op::v0::Constant::create(type, shape, std::vector<float>(ov::shape_size(shape), value));
        };
        auto lstm = std::make_shared<ov::op::v5::LSTMSequence>(
            x, h, c, seq_lengths,
            weight({dirs, 4 * hidden, input}, 0.05f),
            weight({dirs, 4 * hidden, hidden}, 0.025f),
            weight({dirs, 4 * hidden}, 0.01f),
            hidden, direction,
            std::vector<float>{}, std::vector<float>{},
            std::vector<std::string>{"sigmoid", "tanh", "tanh"}, 0.0f);
        return std::make_shared<ov::Model>(lstm->outputs(), ov::ParameterVector{x, h, c}, "artemis_diagnostic");
    };

    for (const auto direction : {ov::op::RecurrentSequenceDirection::FORWARD,
                                 ov::op::RecurrentSequenceDirection::BIDIRECTIONAL}) {
        const char* tag = direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? "bidirectional"
                                                                                         : "unidirectional";
        try {
            auto compiled = core.compile_model(make_model(direction), "GPU");
            log_lstm_implementations(compiled, tag);
        } catch (const std::exception& e) {
            std::cout << "[ARTEMIS_DIAG] " << tag << " compile failed: " << e.what() << std::endl;
        }
    }
    SUCCEED();
}
}  // namespace
