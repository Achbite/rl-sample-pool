#include "store/sample_store.h"

#include <chrono>
#include <cstdlib>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <iomanip>
#include <iostream>
#include <openssl/evp.h>
#include <sstream>
#include <string>
#include <thread>

namespace {

constexpr char kSourceDigest[] =
    "157fba14177a0727abf663c442003e2a5f5c1e297f4af97ea22b45d74cdb32b5";
constexpr char kArtifactDigest[] =
    "71a0f13363d62b5d076c02b00e5b4b269a3e91253b190432f2e83c43cdcf7d3a";
constexpr char kGeneratorIdentity[] =
    "0eb73fc2cb675bdb34bf3db9c99dae62a82f93a5e3a72db84dcf3936464729c8";

int Fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

void Require(bool condition, const std::string& message) {
    if (!condition) std::exit(Fail(message));
}

void SetDigest(rl::common::v1::ContentDigest* digest,
               const std::string& hex) {
    digest->set_algorithm(rl::common::v1::DIGEST_ALGORITHM_SHA256);
    digest->set_hex(hex);
}

std::string Sha256Hex(const std::string& data) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    Require(context != nullptr, "EVP context");
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int size = 0;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(context, data.data(), data.size()) == 1 &&
                    EVP_DigestFinal_ex(context, digest, &size) == 1;
    EVP_MD_CTX_free(context);
    Require(ok && size == 32, "SHA-256 calculation");
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

void RefreshPayloadDigest(rl::training::v1::SampleBatch* batch) {
    batch->clear_payload_digest();
    std::string serialized;
    google::protobuf::io::StringOutputStream output(&serialized);
    google::protobuf::io::CodedOutputStream coded(&output);
    coded.SetSerializationDeterministic(true);
    Require(batch->SerializeToCodedStream(&coded) && !coded.HadError(),
            "deterministic batch serialization");
    coded.Trim();
    SetDigest(batch->mutable_payload_digest(), Sha256Hex(serialized));
}

void FillService(rl::common::v1::ServiceInstanceIdentity* service,
                 const std::string& component,
                 const std::string& instance) {
    service->set_component(component);
    service->set_instance_id(instance);
    service->set_lifecycle_epoch(1);
}

void FillContract(rl::common::v1::ContractIdentity* contract) {
    contract->set_package_name("rl-contracts");
    contract->set_package_version("0.8.0");
    SetDigest(contract->mutable_source_digest(), kSourceDigest);
    SetDigest(contract->mutable_artifact_digest(), kArtifactDigest);
    contract->set_platform("linux/arm64");
    contract->set_generator_identity(kGeneratorIdentity);
}

void FillSchema(rl::common::v1::SchemaIdentity* schema,
                const std::string& id,
                char digest_character) {
    schema->set_schema_id(id);
    schema->set_schema_version(1);
    SetDigest(schema->mutable_canonical_digest(),
              std::string(64, digest_character));
}

rl::training::v1::TrainingSemanticsIdentity MakeSemantics() {
    rl::training::v1::TrainingSemanticsIdentity semantics;
    semantics.set_training_contract_id("maze.training.v3");
    FillSchema(semantics.mutable_observation_schema(),
               "maze.observation.v3", '1');
    FillSchema(semantics.mutable_action_schema(), "maze.action.v1", '2');
    FillSchema(semantics.mutable_reward_schema(), "maze.reward.v3", '3');
    semantics.set_policy_distribution_schema_id("categorical.logits.v1");
    semantics.set_model_architecture_id("maze.mlp-17x64x64.v1");
    SetDigest(semantics.mutable_semantics_digest(), std::string(64, '4'));
    return semantics;
}

rl::training::v1::ModelIdentity MakeModel(uint64_t version) {
    rl::training::v1::ModelIdentity model;
    model.set_model_lineage_id("maze-fixed-map-seed-0");
    model.set_model_version(version);
    SetDigest(model.mutable_artifact_digest(),
              std::string(64, static_cast<char>('a' + version)));
    SetDigest(model.mutable_manifest_digest(),
              std::string(64, static_cast<char>('c' + version)));
    return model;
}

DistributorConfig TestConfig() {
    DistributorConfig config;
    config.max_queue_samples = 10000;
    config.max_queue_fragments = 100;
    config.max_queue_estimated_bytes = 64 * 1024 * 1024;
    config.max_dedup_entries = 1000;
    config.delivery_history_size = 1000;
    config.default_get_timeout_ms = 10;
    config.default_lease_timeout_ms = 100;
    config.contract.package_name = "rl-contracts";
    config.contract.package_version = "0.8.0";
    config.contract.source_digest = kSourceDigest;
    config.contract.artifact_digest = kArtifactDigest;
    config.contract.platform = "linux/arm64";
    config.contract.generator_identity = kGeneratorIdentity;
    return config;
}

rl::training::v1::SampleBatch MakeBatch(
    const std::string& batch_id,
    int sample_count,
    uint64_t sequence,
    uint64_t model_version = 0,
    uint64_t first_step = 0,
    const std::string& producer = "producer-0") {
    rl::training::v1::SampleBatch batch;
    batch.set_batch_id(batch_id);
    batch.set_actor_session_id("actor-session-0");
    batch.set_trajectory_id("trajectory-" + std::to_string(sequence));
    batch.set_actor_id(static_cast<uint32_t>(sequence % 4));
    batch.set_fragment_id(static_cast<uint32_t>(sequence));
    batch.set_fragment_sequence(sequence);
    batch.set_trajectory_end(false);
    batch.set_bootstrap_value(0.25f);
    batch.set_bootstrap_valid(true);
    *batch.mutable_behavior_policy()->mutable_model() = MakeModel(model_version);
    batch.mutable_behavior_policy()->set_distribution_schema_id(
        "categorical.logits.v1");
    SetDigest(batch.mutable_behavior_policy()->mutable_policy_spec_digest(),
              std::string(64, '5'));
    *batch.mutable_training_semantics() = MakeSemantics();
    FillService(batch.mutable_producer(), "aiserver", producer);
    FillContract(batch.mutable_contract());
    batch.set_created_at_unix_ms(1);
    batch.set_first_action_step(first_step);
    batch.set_last_action_step(first_step + sample_count - 1);
    for (int index = 0; index < sample_count; ++index) {
        auto* sample = batch.add_samples();
        sample->add_observation(static_cast<float>(index));
        sample->add_observation(0.25f);
        sample->add_next_observation(static_cast<float>(index + 1));
        sample->add_next_observation(0.5f);
        sample->set_action(index % 9);
        sample->set_reward(0.1f);
        sample->set_old_log_probability(-0.5f);
        sample->set_old_value_prediction(0.2f);
        sample->set_end_kind(
            rl::training::v1::TRANSITION_END_KIND_CONTINUING);
        sample->set_action_step(first_step + index);
    }
    RefreshPayloadDigest(&batch);
    return batch;
}

rl::training::v1::PushSamplesRsp Push(
    SampleStore& store,
    const rl::training::v1::SampleBatch& batch) {
    rl::training::v1::PushSamplesRsp response;
    store.Push(batch, &response);
    return response;
}

void FillConsumer(rl::common::v1::ServiceInstanceIdentity* consumer,
                  const std::string& instance) {
    FillService(consumer, "learner", instance);
}

rl::training::v1::GetBatchRsp Get(
    SampleStore& store,
    int target,
    int timeout_ms,
    int lease_timeout_ms,
    uint64_t version,
    rl::training::v1::BatchSelectionPolicy policy =
        rl::training::v1::BATCH_SELECTION_POLICY_TARGET_ONLY,
    const std::string& consumer = "consumer-0") {
    rl::training::v1::GetBatchReq request;
    FillConsumer(request.mutable_consumer(), consumer);
    request.set_batch_size(target);
    request.set_timeout_ms(timeout_ms);
    request.set_lease_timeout_ms(lease_timeout_ms);
    *request.mutable_target_model() = MakeModel(version);
    request.set_selection_policy(policy);
    *request.mutable_required_semantics() = MakeSemantics();
    rl::training::v1::GetBatchRsp response;
    store.GetBatch(request, &response, []() { return false; });
    return response;
}

rl::training::v1::DeliveryRsp Ack(
    SampleStore& store,
    const std::string& delivery_id,
    rl::training::v1::AckDisposition disposition,
    const std::string& train_update_id = "",
    const std::string& consumer = "consumer-0") {
    rl::training::v1::AckBatchReq request;
    FillConsumer(request.mutable_consumer(), consumer);
    request.set_delivery_id(delivery_id);
    request.set_disposition(disposition);
    request.set_train_update_id(train_update_id);
    rl::training::v1::DeliveryRsp response;
    store.Ack(request, &response);
    return response;
}

void TestPushIdentityAndValidation() {
    DistributorConfig config = TestConfig();
    config.max_queue_samples = 2;
    SampleStore store(config);
    auto first = MakeBatch("batch-1", 2, 1);
    Require(Push(store, first).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "first batch must be accepted");
    Require(Push(store, first).result() ==
                rl::training::v1::PUSH_RESULT_DUPLICATE,
            "same batch_id and payload must be idempotent");
    auto conflict = first;
    conflict.mutable_samples(0)->set_reward(0.2f);
    RefreshPayloadDigest(&conflict);
    Require(Push(store, conflict).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_INVALID,
            "same batch_id with another canonical payload must conflict");
    Require(Push(store, MakeBatch("batch-2", 1, 2)).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_CAPACITY,
            "capacity rejects another unique batch");

    auto invalid_digest = first;
    invalid_digest.set_batch_id("invalid-digest");
    invalid_digest.mutable_payload_digest()->set_hex(std::string(64, '0'));
    Require(Push(store, invalid_digest).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_INVALID,
            "mismatched payload digest is rejected");

    auto wrong_contract = MakeBatch("wrong-contract", 1, 3);
    wrong_contract.mutable_contract()->set_package_version("0.7.0");
    RefreshPayloadDigest(&wrong_contract);
    Require(Push(store, wrong_contract).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY,
            "wrong contract identity fails closed");

    auto wrong_producer = MakeBatch("wrong-producer", 1, 4);
    wrong_producer.mutable_producer()->set_component("rl-aiserver");
    RefreshPayloadDigest(&wrong_producer);
    Require(Push(store, wrong_producer).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY,
            "non-canonical producer component fails closed");

    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.accepted_unique_samples() == 2,
            "accepted count excludes retries and rejects");
    Require(status.duplicate_push_attempt_count() == 1,
            "duplicate attempt count");
    Require(status.rejected_push_attempt_count() == 5,
            "rejected attempt count");
}

void TestTerminalAndBootstrapContract() {
    SampleStore store(TestConfig());
    auto terminal = MakeBatch("terminal", 2, 1);
    auto* final = terminal.mutable_samples(1);
    final->set_terminated(true);
    final->set_end_kind(
        rl::training::v1::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED);
    terminal.set_trajectory_end(true);
    terminal.set_bootstrap_valid(false);
    terminal.set_bootstrap_value(0.0f);
    RefreshPayloadDigest(&terminal);
    Require(Push(store, terminal).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "environment termination carries no bootstrap");

    auto invalid = MakeBatch("invalid-terminal", 1, 2);
    invalid.mutable_samples(0)->set_terminated(true);
    invalid.mutable_samples(0)->set_end_kind(
        rl::training::v1::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED);
    invalid.set_trajectory_end(true);
    RefreshPayloadDigest(&invalid);
    Require(Push(store, invalid).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_INVALID,
            "terminated fragment with bootstrap is rejected");

    auto aborted = MakeBatch("aborted", 1, 3);
    aborted.mutable_samples(0)->set_end_kind(
        rl::training::v1::TRANSITION_END_KIND_PRODUCER_ABORT);
    RefreshPayloadDigest(&aborted);
    Require(Push(store, aborted).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_INVALID,
            "producer abort never enters the pool");
}

void TestBoundedCompletedDedupHistory() {
    DistributorConfig config = TestConfig();
    config.max_dedup_entries = 2;
    SampleStore store(config);
    for (int sequence = 1; sequence <= 3; ++sequence) {
        const std::string id = "completed-" + std::to_string(sequence);
        Require(Push(store, MakeBatch(id, 1, sequence, 0, sequence - 1))
                    .result() == rl::training::v1::PUSH_RESULT_ACCEPTED,
                "batch accepted");
        const auto delivery = Get(store, 1, 10, 100, 0);
        Require(Ack(store, delivery.delivery_id(),
                    rl::training::v1::ACK_DISPOSITION_TRAINED,
                    "update-" + id)
                    .result() == rl::training::v1::DELIVERY_RESULT_APPLIED,
                "completed batch acknowledged");
    }
    Require(Push(store, MakeBatch("completed-3", 1, 3, 0, 2)).result() ==
                rl::training::v1::PUSH_RESULT_DUPLICATE,
            "recent completed retry remains idempotent");
    Require(Push(store, MakeBatch("completed-1", 1, 1, 0, 0)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "bounded completed history eventually evicts old identities");
}

void TestExactPolicyTargetAndAck() {
    SampleStore store(TestConfig());
    Push(store, MakeBatch("v0-a", 60, 1, 0));
    Push(store, MakeBatch("v1-a", 512, 2, 1));
    Push(store, MakeBatch("v0-b", 60, 3, 0, 60));

    auto v0 = Get(store, 100, 20, 100, 0);
    Require(v0.result() == rl::training::v1::GET_BATCH_RESULT_LEASED &&
                v0.actual_batch_size() == 120 && v0.batches_size() == 2,
            "matching policy leases whole fragments only");
    Require(v0.behavior_policy().model().model_version() == 0,
            "response binds the exact behavior policy");
    Require(Ack(store, v0.delivery_id(),
                rl::training::v1::ACK_DISPOSITION_TRAINED, "update-v0")
                .result() == rl::training::v1::DELIVERY_RESULT_APPLIED,
            "trained Ack applies");
    Require(Ack(store, v0.delivery_id(),
                rl::training::v1::ACK_DISPOSITION_TRAINED, "update-v0")
                .result() ==
                rl::training::v1::DELIVERY_RESULT_ALREADY_APPLIED,
            "Ack retry is idempotent");
    Require(Ack(store, v0.delivery_id(),
                rl::training::v1::ACK_DISPOSITION_STALE)
                .result() == rl::training::v1::DELIVERY_RESULT_REJECTED,
            "Ack retry cannot alter disposition");

    auto v1 = Get(store, 512, 20, 100, 1);
    Require(v1.actual_batch_size() == 512 &&
                v1.behavior_policy().model().model_version() == 1,
            "model identity prevents version mixing");
    Ack(store, v1.delivery_id(), rl::training::v1::ACK_DISPOSITION_TRAINED,
        "update-v1");
    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.trained_sample_count() == 632 &&
                status.behavior_versions_size() == 2,
            "status preserves policy-level accounting");
}

void TestMultipleProducersAndDrain() {
    SampleStore store(TestConfig());
    Push(store, MakeBatch("producer-0-fragment", 64, 1));
    Push(store, MakeBatch("producer-1-fragment", 64, 2, 0, 64,
                          "producer-1"));
    auto delivery = Get(store, 128, 20, 100, 0);
    Require(delivery.batches_size() == 2 &&
                delivery.batches(0).producer().instance_id() == "producer-0" &&
                delivery.batches(1).producer().instance_id() == "producer-1",
            "producer identity and FIFO remain distinct");
    Ack(store, delivery.delivery_id(),
        rl::training::v1::ACK_DISPOSITION_SHUTDOWN_UNTRAINED);

    Push(store, MakeBatch("partial", 50, 3, 0, 128));
    Require(Get(store, 512, 10, 100, 0).result() ==
                rl::training::v1::GET_BATCH_RESULT_TIMEOUT,
            "TARGET_ONLY never leases a partial target");
    auto partial = Get(
        store, 512, 10, 100, 0,
        rl::training::v1::BATCH_SELECTION_POLICY_DRAIN_AVAILABLE);
    Require(partial.result() == rl::training::v1::GET_BATCH_RESULT_LEASED &&
                partial.actual_batch_size() == 50,
            "DRAIN_AVAILABLE explicitly leases a partial batch");
    rl::training::v1::NackBatchReq nack;
    FillConsumer(nack.mutable_consumer(), "consumer-0");
    nack.set_delivery_id(partial.delivery_id());
    nack.set_reason("contract test");
    rl::training::v1::DeliveryRsp response;
    store.Nack(nack, &response);
    Require(response.result() == rl::training::v1::DELIVERY_RESULT_APPLIED,
            "Nack requeues the exact payload");
    auto redelivery = Get(
        store, 512, 10, 100, 0,
        rl::training::v1::BATCH_SELECTION_POLICY_DRAIN_AVAILABLE);
    Require(redelivery.batches(0).batch_id() == "partial",
            "redelivery preserves identity");
    Ack(store, redelivery.delivery_id(),
        rl::training::v1::ACK_DISPOSITION_SHUTDOWN_UNTRAINED);
}

void TestRenewExpiryAndSingleConsumer() {
    SampleStore store(TestConfig());
    Push(store, MakeBatch("renew", 12, 1));
    auto first = Get(store, 12, 10, 20, 0);
    rl::training::v1::RenewLeaseReq renew;
    FillConsumer(renew.mutable_consumer(), "consumer-0");
    renew.set_delivery_id(first.delivery_id());
    renew.set_lease_timeout_ms(100);
    rl::training::v1::DeliveryRsp renewal;
    store.RenewLease(renew, &renewal);
    Require(renewal.result() == rl::training::v1::DELIVERY_RESULT_APPLIED,
            "owner renews lease");
    auto busy = Get(store, 12, 10, 100, 0,
                    rl::training::v1::BATCH_SELECTION_POLICY_TARGET_ONLY,
                    "consumer-1");
    Require(busy.result() == rl::training::v1::GET_BATCH_RESULT_BUSY,
            "single-consumer capability is explicit");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    Require(Ack(store, first.delivery_id(),
                rl::training::v1::ACK_DISPOSITION_STALE)
                .result() == rl::training::v1::DELIVERY_RESULT_APPLIED,
            "renewed lease remains valid");

    Push(store, MakeBatch("expire", 12, 2, 0, 20));
    auto expiring = Get(store, 12, 10, 20, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    auto redelivery = Get(store, 12, 10, 100, 0);
    Require(redelivery.result() == rl::training::v1::GET_BATCH_RESULT_LEASED &&
                redelivery.delivery_id() != expiring.delivery_id(),
            "expired lease is redelivered with a new delivery identity");
    Ack(store, redelivery.delivery_id(),
        rl::training::v1::ACK_DISPOSITION_INVALID);

    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.lease_renew_count() == 1 &&
                status.expired_lease_count() == 1 &&
                status.stale_sample_count() == 12 &&
                status.invalid_sample_count() == 12 &&
                status.consumer_busy_count() == 1,
            "lease and disposition accounting closes exactly");
    Require(status.contract().package_version() == "0.8.0" &&
                status.distributor().component() == "sample-pool" &&
                status.ready() && status.ingress_ready() &&
                status.pool_ready(),
            "status exposes exact contract and service identity");
}

}  // namespace

int main() {
    TestPushIdentityAndValidation();
    TestTerminalAndBootstrapContract();
    TestBoundedCompletedDedupHistory();
    TestExactPolicyTargetAndAck();
    TestMultipleProducersAndDrain();
    TestRenewExpiryAndSingleConsumer();
    std::cout << "sample_store_contract: PASS" << std::endl;
    return 0;
}
