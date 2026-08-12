// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "subgraphs_builders.hpp"
#include "common_test_utils/common_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/op_conversions/bidirectional_sequences_decomposition.hpp"
#include "transformations/op_conversions/convert_sequences_to_tensor_iterator.hpp"

namespace {

class SerializeBaseTest : virtual public ov::test::SubgraphBaseStaticTest {
public:
    std::string cacheDirName;

    void run() override {
        cacheDirName = ov::test::utils::generateTestFilePrefix() + "_cache_serialize";
        std::stringstream model_stream;
        ov::AnyMap config = { ov::cache_dir(cacheDirName) };
        compiledModel = core->compile_model(function, targetDevice);
        compiledModel.export_model(model_stream);
        auto imported_model = core->import_model(model_stream, targetDevice);
    }

    void TearDown() override{
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);
        ov::test::SubgraphBaseStaticTest::TearDown();
    }
};


class LSTMSequenceTest : virtual public SerializeBaseTest {
public:
    void SetUp() override {
        constexpr size_t batch_size = 2;
        constexpr size_t sequence_length = 3;
        constexpr size_t input_size = 4;
        constexpr size_t hidden_size = 5;
        constexpr size_t num_directions = 2;
        const auto model_type = ov::element::f16;

        targetDevice = "GPU";
        abs_threshold = 0.f;
        rel_threshold = 0.f;
        const std::vector<ov::Shape> input_shapes{
            {batch_size, sequence_length, input_size},
            {batch_size, num_directions, hidden_size},
            {batch_size, num_directions, hidden_size},
        };
        init_input_shapes(ov::test::static_shapes_to_test_representation(input_shapes));

        auto x = std::make_shared<ov::op::v0::Parameter>(model_type, input_shapes[0]);
        auto initial_hidden = std::make_shared<ov::op::v0::Parameter>(model_type, input_shapes[1]);
        auto initial_cell = std::make_shared<ov::op::v0::Parameter>(model_type, input_shapes[2]);
        auto sequence_lengths =
            ov::op::v0::Constant::create(ov::element::i64, {batch_size}, {sequence_length, sequence_length});
        auto weights = ov::op::v0::Constant::create(
            model_type,
            {num_directions, 4 * hidden_size, input_size},
            std::vector<float>(num_directions * 4 * hidden_size * input_size, 0.05f));
        auto recurrence_weights = ov::op::v0::Constant::create(
            model_type,
            {num_directions, 4 * hidden_size, hidden_size},
            std::vector<float>(num_directions * 4 * hidden_size * hidden_size, 0.025f));
        auto biases = ov::op::v0::Constant::create(model_type,
                                                   {num_directions, 4 * hidden_size},
                                                   std::vector<float>(num_directions * 4 * hidden_size, 0.01f));

        auto lstm = std::make_shared<ov::op::v5::LSTMSequence>(
            x,
            initial_hidden,
            initial_cell,
            sequence_lengths,
            weights,
            recurrence_weights,
            biases,
            hidden_size,
            ov::op::RecurrentSequenceDirection::BIDIRECTIONAL);
        lstm->set_friendly_name("serialize_bidirectional_lstm");
        function = std::make_shared<ov::Model>(lstm->outputs(),
                                               ov::ParameterVector{x, initial_hidden, initial_cell},
                                               "BidirectionalLSTMSequenceSerialize");
    }

    void run() override {
        cacheDirName = ov::test::utils::generateTestFilePrefix() + "_cache_serialize";
        ov::test::utils::removeFilesWithExt(cacheDirName, "blob");
        ov::test::utils::removeFilesWithExt(cacheDirName, "cl_cache");
        ov::test::utils::removeDir(cacheDirName);

        ov::AnyMap config;

        generate_inputs(targetStaticShapes.front());
        compiledModel = core->compile_model(function, targetDevice, config);
        const auto original_outputs = infer_model(compiledModel);

        std::stringstream model_stream;
        compiledModel.export_model(model_stream);
        auto imported_model = core->import_model(model_stream, targetDevice, config);
        compare(original_outputs, infer_model(imported_model));

        config.insert(ov::cache_dir(cacheDirName));
        auto cache_populating_model = core->compile_model(function, targetDevice, config);
        compare(original_outputs, infer_model(cache_populating_model));
        auto cache_loaded_model = core->compile_model(function, targetDevice, config);
        compare(original_outputs, infer_model(cache_loaded_model));

        // GPU-to-GPU parity alone would accept a consistently-wrong native
        // result, so also compare against an explicitly decomposed reference
        // graph with numeric (non-exact) thresholds.
        auto reference_function = function->clone();
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::BidirectionalLSTMSequenceDecomposition>();
        manager.register_pass<ov::pass::ConvertLSTMSequenceToTensorIterator>();
        manager.run_passes(reference_function);
        auto reference_model = core->compile_model(reference_function, targetDevice);
        abs_threshold = 0.005f;
        rel_threshold = 0.03f;
        compare(original_outputs, infer_model(reference_model));
        abs_threshold = 0.f;
        rel_threshold = 0.f;
    }

private:
    std::vector<ov::Tensor> infer_model(ov::CompiledModel& model) {
        auto request = model.create_infer_request();
        const auto& parameters = function->get_parameters();
        for (size_t i = 0; i < parameters.size(); ++i)
            request.set_input_tensor(i, inputs.at(parameters[i]));
        request.infer();

        std::vector<ov::Tensor> outputs;
        for (size_t i = 0; i < function->get_output_size(); ++i)
            outputs.push_back(request.get_output_tensor(i));
        return outputs;
    }
};


class GRUSequenceTest : virtual public SerializeBaseTest {
public:
    void SetUp() override {
        std::string cacheDirName = "cache_gru";
        auto init_shape = ov::PartialShape({1, 30, 512});
        auto batch_size = static_cast<size_t>(init_shape[0].get_length());
        size_t hidden_size = 10;
        size_t sequence_axis = 1;
        targetDevice = "GPU";
        auto input_size = static_cast<size_t>(init_shape[init_shape.size()-1].get_length());
        function = tests::makeLBRGRUSequence(ov::element::f16, init_shape, batch_size, input_size, hidden_size, sequence_axis,
            ov::op::RecurrentSequenceDirection::FORWARD);
    }
};

TEST_F(LSTMSequenceTest, smoke_BidirectionalExportImportCacheInferenceParity) {
    run();
}

TEST_F(GRUSequenceTest, smoke_serialize) {
    run();
}

class GpuCacheDirWithDotsParamTest : public ::testing::TestWithParam<std::string> {
protected:
    ov::Core core;
    std::string cacheDir;

    void SetUp() override {
        cacheDir = ov::test::utils::generateTestFilePrefix() + GetParam();

        // Clean previous
        ov::test::utils::removeFilesWithExt(cacheDir, "blob");
        ov::test::utils::removeFilesWithExt(cacheDir, "cl_cache");
        ov::test::utils::removeDir(cacheDir);

        core.set_property(ov::cache_dir(cacheDir));
    }

    void TearDown() override {
        ov::test::utils::removeFilesWithExt(cacheDir, "blob");
        ov::test::utils::removeFilesWithExt(cacheDir, "cl_cache");
        ov::test::utils::removeDir(cacheDir);
    }
};

TEST_P(GpuCacheDirWithDotsParamTest, smoke_PopulateAndReuseCache) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 8, 8});
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto res = std::make_shared<ov::op::v0::Result>(relu);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{param}, "CacheDotsModel");
    core.compile_model(model, "GPU");
}

INSTANTIATE_TEST_SUITE_P(CacheDirDotVariants,
                         GpuCacheDirWithDotsParamTest,
                         ::testing::Values("/test_encoder/test_encoder.encrypted/", "/test_encoder/test_encoder.encrypted"));

}  // namespace
