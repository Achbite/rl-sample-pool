#include "store/sample_pool_coordinator.h"

#include <cstdlib>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <iostream>
#include <map>
#include <openssl/evp.h>
#include <set>
#include <sstream>
#include <string>

namespace {

constexpr char kSourceDigest[] =
    "1111111111111111111111111111111111111111111111111111111111111111";
constexpr char kArtifactDigest[] =
    "2222222222222222222222222222222222222222222222222222222222222222";
constexpr char kGeneratorIdentity[] =
    "3333333333333333333333333333333333333333333333333333333333333333";
constexpr char kTrainingContractDigest[] =
    "4444444444444444444444444444444444444444444444444444444444444444";

void Require(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
}

void SetDigest(rl::common::v1::ContentDigest* digest,
               const std::string& hex) {
    digest->set_algorithm(rl::common::v1::DIGEST_ALGORITHM_SHA256);
    digest->set_hex(hex);
}

std::string Sha256Hex(const std::string& data) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    Require(context != nullptr, "create SHA-256 context");
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    const bool ok =
        EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
        EVP_DigestUpdate(context, data.data(), data.size()) == 1 &&
        EVP_DigestFinal_ex(context, digest, &digest_size) == 1;
    EVP_MD_CTX_free(context);
    Require(ok && digest_size == 32, "compute SHA-256");

    static constexpr char kHex[] = "0123456789abcdef";
    std::string result;
    result.reserve(64);
    for (unsigned int index = 0; index < digest_size; ++index) {
        result.push_back(kHex[digest[index] >> 4]);
        result.push_back(kHex[digest[index] & 0x0f]);
    }
    return result;
}

std::string DeterministicBytes(
    const rl::training::v1::ProcessedTransitionEnvelope& envelope) {
    std::string serialized;
    google::protobuf::io::StringOutputStream output(&serialized);
    google::protobuf::io::CodedOutputStream coded(&output);
    coded.SetSerializationDeterministic(true);
    Require(envelope.SerializeToCodedStream(&coded) && !coded.HadError(),
            "serialize envelope deterministically");
    coded.Trim();
    return serialized;
}

void FillService(rl::common::v1::ServiceInstanceIdentity* identity,
                 const std::string& component,
                 const std::string& instance_id) {
    identity->set_component(component);
    identity->set_instance_id(instance_id);
    identity->set_lifecycle_epoch(1);
}

SamplePoolConfig MakeConfig() {
    SamplePoolConfig config;
    config.backend_type = "local_memory";
    config.capacity_transitions = 16;
    config.capacity_bytes = 1024 * 1024;
    config.sampling_seed = 7;
    config.max_dedup_entries = 64;
    config.high_watermark_ratio = 0.8;
    config.default_get_timeout_ms = 10;
    config.default_lease_timeout_ms = 1000;
    config.delivery_history_size = 16;
    config.contract.package_name = "rl-contracts";
    config.contract.package_version = "0.15.0";
    config.contract.source_digest = kSourceDigest;
    config.contract.artifact_digest = kArtifactDigest;
    config.contract.platform = "linux/arm64";
    config.contract.generator_identity = kGeneratorIdentity;
    return config;
}

void FillBehaviorModel(rl::training::v1::ModelIdentity* model) {
    model->set_model_lineage_id("lineage-fixed");
    model->set_model_step(3);
    SetDigest(model->mutable_artifact_digest(), std::string(64, 'a'));
    SetDigest(model->mutable_manifest_digest(), std::string(64, 'b'));
}

rl::training::v1::ProcessedTransitionEnvelope MakeEnvelope() {
    rl::training::v1::ProcessedTransitionEnvelope envelope;
    envelope.set_envelope_id("envelope-fixed");
    FillService(envelope.mutable_producer(),
                "sample-distributor", "sample-distributor-fixed");
    SetDigest(envelope.mutable_training_contract_digest(),
              kTrainingContractDigest);
    FillBehaviorModel(envelope.mutable_behavior_model());

    for (uint32_t index = 0; index < 2; ++index) {
        auto* transition = envelope.add_samples();
        transition->set_item_id("item-" + std::to_string(index));
        transition->add_observation(static_cast<float>(index));
        transition->add_observation(0.25f);
        transition->set_action(static_cast<int32_t>(index + 2));
        transition->set_behavior_log_probability(-0.5f - index * 0.1f);
        transition->set_behavior_value(0.2f + index * 0.1f);
        transition->set_advantage(index == 0 ? 0.5f : -0.25f);
        transition->set_value_target(index == 0 ? 0.7f : 0.05f);
        transition->set_behavior_model_step(3);
        transition->set_created_at_unix_ms(1700000000000 + index);
    }

    envelope.clear_payload_digest();
    const std::string payload_digest =
        Sha256Hex(DeterministicBytes(envelope));
    SetDigest(envelope.mutable_payload_digest(), payload_digest);
    return envelope;
}

void TestPushGetAck() {
    SamplePoolCoordinator pool(MakeConfig());
    const auto envelope = MakeEnvelope();

    rl::training::v1::PushSamplesReq push_request;
    *push_request.mutable_envelope() = envelope;
    rl::training::v1::PushSamplesRsp push_response;
    pool.Push(push_request, &push_response);
    Require(push_response.result() == rl::training::v1::PUSH_RESULT_ACCEPTED &&
                push_response.envelope_id() == envelope.envelope_id() &&
                push_response.payload_digest().SerializeAsString() ==
                    envelope.payload_digest().SerializeAsString(),
            "Push accepts the fixed processed-transition envelope");

    rl::training::v1::GetBatchReq get_request;
    get_request.set_requested_transitions(2);
    get_request.set_timeout_ms(10);
    get_request.set_lease_timeout_ms(1000);
    FillService(get_request.mutable_consumer(), "learner", "learner-fixed");
    SetDigest(get_request.mutable_required_training_contract_digest(),
              kTrainingContractDigest);
    rl::training::v1::GetBatchRsp get_response;
    pool.GetBatch(get_request, &get_response, []() { return false; });

    std::map<std::string, std::string> expected_transitions;
    for (const auto& transition : envelope.samples()) {
        expected_transitions.emplace(
            transition.item_id(), transition.SerializeAsString());
    }
    std::map<std::string, std::string> leased_transitions;
    for (const auto& item : get_response.items()) {
        leased_transitions.emplace(
            item.transition().item_id(),
            item.transition().SerializeAsString());
    }
    Require(get_response.result() ==
                rl::training::v1::GET_BATCH_RESULT_LEASED &&
                get_response.items_size() == 2 &&
                leased_transitions == expected_transitions,
            "Get leases the two fixed transitions unchanged and uniquely");

    rl::training::v1::AckBatchReq ack_request;
    FillService(ack_request.mutable_consumer(), "learner", "learner-fixed");
    ack_request.set_delivery_id(get_response.delivery_id());
    ack_request.set_disposition(rl::training::v1::ACK_DISPOSITION_TRAINED);
    ack_request.set_train_update_id("update-fixed");
    rl::training::v1::DeliveryRsp ack_response;
    pool.Ack(ack_request, &ack_response);
    Require(ack_response.result() ==
                rl::training::v1::DELIVERY_RESULT_APPLIED,
            "Ack settles the leased fixed transitions as trained");

    rl::training::v1::SamplePoolStatusReq status_request;
    rl::training::v1::SamplePoolStatusRsp status_response;
    pool.GetStatus(status_request, &status_response);
    Require(status_response.leased_transitions() == 0 &&
                status_response.resident_transitions() == 0,
            "the trained transitions no longer belong to the delivery");
}

}  // namespace

int main() {
    TestPushGetAck();
    std::cout << "sample_pool_development_contract: PASS" << std::endl;
    return 0;
}
