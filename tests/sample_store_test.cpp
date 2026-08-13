#include "store/sample_store.h"

#include <algorithm>
#include <atomic>
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
    "7e3eb7227e67a2a880130c9c82f87041691c0095f838a60f80abc1f387c1c5b3";
constexpr char kArtifactDigest[] =
    "077ac6d61486fafd5f0430eeb05a492764b36e073282f6d7626d0414bb5b2ddf";
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
    contract->set_package_version("0.11.0");
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
    FillSchema(semantics.mutable_reward_schema(), "maze.reward.v4", '3');
    semantics.set_policy_distribution_schema_id("categorical.logits.v1");
    semantics.set_model_architecture_id("maze.mlp-17x64x64.v1");
    SetDigest(semantics.mutable_semantics_digest(), std::string(64, '4'));
    return semantics;
}

int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
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
    config.max_fragment_samples = 128;
    config.contract.package_name = "rl-contracts";
    config.contract.package_version = "0.11.0";
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
    const std::string& producer = "producer-0",
    int64_t created_at_unix_ms = 0) {
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
    batch.mutable_behavior_policy()->set_model_lineage_id(
        "maze-fixed-map-seed-0");
    batch.mutable_behavior_policy()->set_model_version(model_version);
    batch.mutable_behavior_policy()->set_distribution_schema_id(
        "categorical.logits.v1");
    SetDigest(batch.mutable_behavior_policy()->mutable_policy_spec_digest(),
              std::string(64, '5'));
    *batch.mutable_training_semantics() = MakeSemantics();
    FillService(batch.mutable_producer(), "aiserver", producer);
    FillContract(batch.mutable_contract());
    batch.set_created_at_unix_ms(created_at_unix_ms > 0
                                     ? created_at_unix_ms
                                     : NowMs());
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

rl::training::v1::UpsertSampleDemandReq MakeDemand(
    uint64_t epoch = 1,
    int64_t max_buffered_samples = 10000,
    int64_t max_buffered_fragments = 100,
    uint64_t reference_model_version = 1000,
    uint32_t max_version_lag = 1000,
    int64_t ttl_ms = 10000) {
    rl::training::v1::UpsertSampleDemandReq request;
    auto* demand = request.mutable_demand();
    demand->set_demand_id("test-demand");
    demand->set_demand_epoch(epoch);
    FillService(demand->mutable_consumer(), "learner", "consumer-0");
    FillContract(demand->mutable_contract());
    *demand->mutable_training_semantics() = MakeSemantics();
    demand->mutable_freshness()->set_model_lineage_id(
        "maze-fixed-map-seed-0");
    demand->mutable_freshness()->set_reference_model_version(
        reference_model_version);
    demand->mutable_freshness()->set_max_version_lag(max_version_lag);
    demand->mutable_freshness()->set_max_sample_age_ms(60000);
    demand->mutable_freshness()->set_distribution_schema_id(
        "categorical.logits.v1");
    SetDigest(demand->mutable_freshness()->mutable_policy_spec_digest(),
              std::string(64, '5'));
    demand->mutable_assembly()->set_target_samples(1);
    demand->mutable_assembly()->set_max_samples(128);
    demand->mutable_assembly()->set_mode(
        rl::training::v1::BATCH_ASSEMBLY_MODE_TARGET_BOUNDED);
    demand->set_max_buffered_samples(max_buffered_samples);
    demand->set_max_buffered_fragments(max_buffered_fragments);
    demand->set_max_buffered_estimated_bytes(64 * 1024 * 1024);
    demand->set_expires_at_unix_ms(NowMs() + ttl_ms);
    return request;
}

rl::training::v1::AcquireSampleCreditReq MakeCreditRequest(
    const rl::training::v1::SampleBatch& batch,
    const std::string& request_id = "") {
    rl::training::v1::AcquireSampleCreditReq request;
    request.set_request_id(request_id.empty()
                               ? batch.batch_id() + "-credit-request"
                               : request_id);
    *request.mutable_producer() = batch.producer();
    *request.mutable_contract() = batch.contract();
    request.set_batch_id(batch.batch_id());
    *request.mutable_payload_digest() = batch.payload_digest();
    *request.mutable_behavior_policy() = batch.behavior_policy();
    *request.mutable_training_semantics() = batch.training_semantics();
    request.set_sample_count(batch.samples_size());
    request.set_fragment_count(1);
    request.set_estimated_bytes(batch.ByteSizeLong());
    request.set_created_at_unix_ms(batch.created_at_unix_ms());
    return request;
}

rl::training::v1::SampleDemandRsp Upsert(
    SampleStore& store,
    const rl::training::v1::UpsertSampleDemandReq& request) {
    rl::training::v1::SampleDemandRsp response;
    store.UpsertDemand(request, &response);
    return response;
}

rl::training::v1::SampleCreditGrant Acquire(
    SampleStore& store,
    const rl::training::v1::AcquireSampleCreditReq& request) {
    rl::training::v1::SampleCreditGrant response;
    store.AcquireCredit(request, &response);
    return response;
}

rl::training::v1::ReleaseSampleCreditRsp ReleaseCredit(
    SampleStore& store,
    const rl::training::v1::SampleCreditGrant& credit,
    const rl::training::v1::SampleBatch& batch) {
    rl::training::v1::ReleaseSampleCreditReq request;
    *request.mutable_producer() = batch.producer();
    *request.mutable_contract() = batch.contract();
    request.set_credit_id(credit.credit_id());
    request.set_batch_id(batch.batch_id());
    *request.mutable_payload_digest() = batch.payload_digest();
    request.set_reason(
        rl::training::v1::SAMPLE_CREDIT_RELEASE_REASON_PRODUCER_ABORT);
    rl::training::v1::ReleaseSampleCreditRsp response;
    store.ReleaseCredit(request, &response);
    return response;
}

rl::training::v1::PushSamplesRsp Push(
    SampleStore& store,
    const rl::training::v1::SampleBatch& batch) {
    rl::training::v1::UpsertSampleDemandReq demand_request;
    auto* demand = demand_request.mutable_demand();
    demand->set_demand_id("test-demand");
    demand->set_demand_epoch(1);
    FillService(demand->mutable_consumer(), "learner", "consumer-0");
    FillContract(demand->mutable_contract());
    *demand->mutable_training_semantics() = MakeSemantics();
    demand->mutable_freshness()->set_model_lineage_id(
        "maze-fixed-map-seed-0");
    demand->mutable_freshness()->set_reference_model_version(1000);
    demand->mutable_freshness()->set_max_version_lag(1000);
    demand->mutable_freshness()->set_max_sample_age_ms(60000);
    demand->mutable_freshness()->set_distribution_schema_id(
        "categorical.logits.v1");
    SetDigest(demand->mutable_freshness()->mutable_policy_spec_digest(),
              std::string(64, '5'));
    demand->mutable_assembly()->set_target_samples(1);
    demand->mutable_assembly()->set_max_samples(128);
    demand->mutable_assembly()->set_mode(
        rl::training::v1::BATCH_ASSEMBLY_MODE_TARGET_BOUNDED);
    demand->set_max_buffered_samples(10000);
    demand->set_max_buffered_fragments(100);
    demand->set_max_buffered_estimated_bytes(64 * 1024 * 1024);
    demand->set_expires_at_unix_ms(NowMs() + 10000);
    rl::training::v1::SampleDemandRsp demand_response;
    store.UpsertDemand(demand_request, &demand_response);

    rl::training::v1::AcquireSampleCreditReq credit_request;
    credit_request.set_request_id(batch.batch_id() + "-credit-request");
    *credit_request.mutable_producer() = batch.producer();
    *credit_request.mutable_contract() = batch.contract();
    credit_request.set_batch_id(batch.batch_id());
    *credit_request.mutable_payload_digest() = batch.payload_digest();
    *credit_request.mutable_behavior_policy() = batch.behavior_policy();
    *credit_request.mutable_training_semantics() = batch.training_semantics();
    credit_request.set_sample_count(batch.samples_size());
    credit_request.set_fragment_count(1);
    credit_request.set_estimated_bytes(batch.ByteSizeLong());
    credit_request.set_created_at_unix_ms(batch.created_at_unix_ms());
    rl::training::v1::SampleCreditGrant credit;
    store.AcquireCredit(credit_request, &credit);
    rl::training::v1::PushSamplesRsp response;
    if (credit.result() != rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED) {
        response.set_ret_code(-1);
        response.set_message(credit.message());
        response.set_batch_id(batch.batch_id());
        if (credit.result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_WAIT_CAPACITY ||
            credit.result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_WAIT_INFLIGHT_LIMIT) {
            response.set_result(
                rl::training::v1::PUSH_RESULT_REJECTED_CAPACITY);
        } else if (credit.result() ==
                   rl::training::v1::SAMPLE_CREDIT_RESULT_REJECTED_IDENTITY) {
            response.set_result(
                rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY);
        } else {
            response.set_result(
                rl::training::v1::PUSH_RESULT_REJECTED_INVALID);
        }
        return response;
    }
    rl::training::v1::PushSamplesReq request;
    request.set_credit_id(credit.credit_id());
    *request.mutable_batch() = batch;
    store.Push(request, &response);
    return response;
}

void FillConsumer(rl::common::v1::ServiceInstanceIdentity* consumer,
                  const std::string& instance) {
    FillService(consumer, "learner", instance);
}

rl::training::v1::GetBatchReq MakeGetBatchRequest(
    int target,
    int timeout_ms,
    int lease_timeout_ms,
    uint64_t version,
    rl::training::v1::BatchAssemblyMode mode =
        rl::training::v1::BATCH_ASSEMBLY_MODE_TARGET_BOUNDED,
    const std::string& consumer = "consumer-0") {
    rl::training::v1::GetBatchReq request;
    FillConsumer(request.mutable_consumer(), consumer);
    request.mutable_assembly()->set_target_samples(target);
    request.mutable_assembly()->set_max_samples(target + 127);
    request.mutable_assembly()->set_mode(mode);
    request.set_timeout_ms(timeout_ms);
    request.set_lease_timeout_ms(lease_timeout_ms);
    request.mutable_freshness()->set_model_lineage_id(
        "maze-fixed-map-seed-0");
    request.mutable_freshness()->set_reference_model_version(version);
    request.mutable_freshness()->set_max_version_lag(1);
    request.mutable_freshness()->set_max_sample_age_ms(60000);
    request.mutable_freshness()->set_distribution_schema_id(
        "categorical.logits.v1");
    SetDigest(request.mutable_freshness()->mutable_policy_spec_digest(),
              std::string(64, '5'));
    *request.mutable_required_semantics() = MakeSemantics();
    return request;
}

rl::training::v1::GetBatchRsp Get(
    SampleStore& store,
    int target,
    int timeout_ms,
    int lease_timeout_ms,
    uint64_t version,
    rl::training::v1::BatchAssemblyMode mode =
        rl::training::v1::BATCH_ASSEMBLY_MODE_TARGET_BOUNDED,
    const std::string& consumer = "consumer-0") {
    const auto request = MakeGetBatchRequest(
        target, timeout_ms, lease_timeout_ms, version, mode, consumer);
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
    const auto first_response = Push(store, first);
    Require(first_response.result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "first batch must be accepted: " + first_response.message());
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

    SampleStore validation_store(TestConfig());
    auto invalid_digest = first;
    invalid_digest.set_batch_id("invalid-digest");
    invalid_digest.mutable_payload_digest()->set_hex(std::string(64, '0'));
    Require(Push(validation_store, invalid_digest).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_INVALID,
            "mismatched payload digest is rejected");

    auto wrong_contract = MakeBatch("wrong-contract", 1, 3);
    wrong_contract.mutable_contract()->set_package_version("0.7.0");
    RefreshPayloadDigest(&wrong_contract);
    Require(Push(validation_store, wrong_contract).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY,
            "wrong contract identity fails closed");

    auto wrong_producer = MakeBatch("wrong-producer", 1, 4);
    wrong_producer.mutable_producer()->set_component("rl-aiserver");
    RefreshPayloadDigest(&wrong_producer);
    Require(Push(validation_store, wrong_producer).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY,
            "non-canonical producer component fails closed");

    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.accepted_unique_samples() == 2,
            "accepted count excludes retries and rejects");
    Require(status.duplicate_push_attempt_count() == 1,
            "duplicate attempt count");
    Require(status.rejected_push_attempt_count() == 0,
            "credit rejections do not masquerade as Push rejections");
    Require(status.credit_grant_count() == 1 &&
                status.credit_commit_count() == 1 &&
                status.credit_wait_capacity_count() == 1,
            "credit reservation and capacity accounting are explicit");
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
    const int64_t created_at = NowMs();
    for (int sequence = 1; sequence <= 3; ++sequence) {
        const std::string id = "completed-" + std::to_string(sequence);
        Require(Push(store, MakeBatch(id, 1, sequence, 0, sequence - 1,
                                      "producer-0", created_at))
                    .result() == rl::training::v1::PUSH_RESULT_ACCEPTED,
                "batch accepted");
        const auto delivery = Get(store, 1, 10, 100, 0);
        Require(Ack(store, delivery.delivery_id(),
                    rl::training::v1::ACK_DISPOSITION_TRAINED,
                    "update-" + id)
                    .result() == rl::training::v1::DELIVERY_RESULT_APPLIED,
                "completed batch acknowledged");
    }
    Require(Push(store, MakeBatch("completed-3", 1, 3, 0, 2,
                                  "producer-0", created_at)).result() ==
                rl::training::v1::PUSH_RESULT_DUPLICATE,
            "recent completed retry remains idempotent");
    Require(Push(store, MakeBatch("completed-1", 1, 1, 0, 0,
                                  "producer-0", created_at)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "bounded completed history eventually evicts old identities");
}

void TestBoundedMultiVersionAssemblyAndAck() {
    SampleStore store(TestConfig());
    Push(store, MakeBatch("v0-a", 60, 1, 0));
    Push(store, MakeBatch("v1-a", 128, 2, 1));
    Push(store, MakeBatch("v1-b", 128, 3, 1, 128));
    Push(store, MakeBatch("v1-c", 128, 4, 1, 256));
    Push(store, MakeBatch("v0-b", 76, 5, 0, 60));

    auto mixed = Get(store, 512, 20, 100, 1);
    Require(mixed.result() == rl::training::v1::GET_BATCH_RESULT_LEASED &&
                mixed.actual_batch_size() == 520 &&
                mixed.actual_batch_size() <= 639 &&
                mixed.batches_size() == 5,
            "bounded assembly reaches target with whole fragments");
    Require(mixed.minimum_behavior_model_version() == 0 &&
                mixed.maximum_behavior_model_version() == 1,
            "compatible policy versions may share one PPO delivery");
    Require(Ack(store, mixed.delivery_id(),
                rl::training::v1::ACK_DISPOSITION_TRAINED, "update-v0")
                .result() == rl::training::v1::DELIVERY_RESULT_APPLIED,
            "trained Ack applies");
    Require(Ack(store, mixed.delivery_id(),
                rl::training::v1::ACK_DISPOSITION_TRAINED, "update-v0")
                .result() ==
                rl::training::v1::DELIVERY_RESULT_ALREADY_APPLIED,
            "Ack retry is idempotent");
    Require(Ack(store, mixed.delivery_id(),
                rl::training::v1::ACK_DISPOSITION_STALE)
                .result() == rl::training::v1::DELIVERY_RESULT_REJECTED,
            "Ack retry cannot alter disposition");

    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.trained_sample_count() == 520 &&
                status.behavior_versions_size() == 2,
            "status preserves policy-level accounting");
}

void TestFreshnessRejectsOldButRetainsFuture() {
    SampleStore store(TestConfig());
    Push(store, MakeBatch("stale-version", 64, 1, 0, 0, "producer-0",
                          NowMs()));
    Push(store, MakeBatch("current-version", 64, 2, 2, 64, "producer-0",
                          NowMs()));
    Push(store, MakeBatch("future-version", 64, 3, 3, 128, "producer-0",
                          NowMs()));

    auto current = Get(store, 64, 20, 100, 2);
    Require(current.result() == rl::training::v1::GET_BATCH_RESULT_LEASED &&
                current.minimum_behavior_model_version() == 2 &&
                current.maximum_behavior_model_version() == 2,
            "version window expires old samples and excludes future samples");
    Ack(store, current.delivery_id(),
        rl::training::v1::ACK_DISPOSITION_TRAINED, "update-current");

    auto future = Get(store, 64, 20, 100, 3);
    Require(future.result() == rl::training::v1::GET_BATCH_RESULT_LEASED &&
                future.minimum_behavior_model_version() == 3,
            "future samples remain available when reference catches up");
    Ack(store, future.delivery_id(),
        rl::training::v1::ACK_DISPOSITION_TRAINED, "update-future");

    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.stale_sample_count() == 64,
            "expired ready samples have an explicit stale disposition");
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
        rl::training::v1::BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE);
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
        rl::training::v1::BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE);
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
                    rl::training::v1::BATCH_ASSEMBLY_MODE_TARGET_BOUNDED,
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
    Require(status.contract().package_version() == "0.11.0" &&
                status.distributor().component() == "sample-distributor" &&
                status.ready() && status.ingress_ready() &&
                status.pool_ready(),
            "status exposes exact contract and service identity");
}

void TestDemandCreditFlowControl() {
    DistributorConfig config = TestConfig();
    config.max_queue_samples = 256;
    config.credit_ttl_ms = 10;
    SampleStore store(config);

    auto batch_a = MakeBatch("credit-a", 128, 1, 2);
    auto request_a = MakeCreditRequest(batch_a, "request-a");
    Require(Acquire(store, request_a).result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_WAIT_NO_DEMAND,
            "producer waits when no Learner demand exists");

    auto demand = MakeDemand(1, 128, 1, 2, 1);
    Require(Upsert(store, demand).result() ==
                rl::training::v1::SAMPLE_DEMAND_RESULT_APPLIED,
            "Learner demand is applied");
    const auto credit_a = Acquire(store, request_a);
    Require(credit_a.result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED &&
                credit_a.state() ==
                    rl::training::v1::SAMPLE_CREDIT_STATE_RESERVED,
            "credit reserves the declared demand window");

    auto batch_b = MakeBatch("credit-b", 1, 2, 2, 128);
    auto request_b = MakeCreditRequest(batch_b, "request-b");
    Require(Acquire(store, request_b).result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_WAIT_INFLIGHT_LIMIT,
            "a full demand window is a retryable wait");

    rl::training::v1::ReleaseSampleCreditReq release;
    *release.mutable_producer() = batch_a.producer();
    *release.mutable_contract() = batch_a.contract();
    release.set_credit_id(credit_a.credit_id());
    release.set_batch_id(batch_a.batch_id());
    *release.mutable_payload_digest() = batch_a.payload_digest();
    release.set_reason(
        rl::training::v1::SAMPLE_CREDIT_RELEASE_REASON_PRODUCER_ABORT);
    rl::training::v1::ReleaseSampleCreditRsp release_response;
    store.ReleaseCredit(release, &release_response);
    Require(release_response.state() ==
                rl::training::v1::SAMPLE_CREDIT_STATE_RELEASED &&
                Acquire(store, request_b).result() ==
                    rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED,
            "released reservation immediately returns capacity");

    auto epoch_two = MakeDemand(2, 256, 2, 2, 1);
    Require(Upsert(store, epoch_two).result() ==
                rl::training::v1::SAMPLE_DEMAND_RESULT_APPLIED,
            "a newer demand epoch replaces the old window");
    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.reserved_samples() == 0 &&
                status.reserved_fragments() == 0 &&
                status.credit_revoke_count() >= 1,
            "epoch replacement revokes every outstanding reservation");

    auto immutable_conflict = epoch_two;
    immutable_conflict.mutable_demand()->set_max_buffered_samples(512);
    Require(Upsert(store, immutable_conflict).result() ==
                rl::training::v1::SAMPLE_DEMAND_RESULT_REJECTED_INVALID,
            "an epoch cannot mutate its immutable demand window");

    auto batch_c = MakeBatch("credit-c", 1, 3, 2, 129);
    auto request_c = MakeCreditRequest(batch_c, "request-c");
    const auto credit_c = Acquire(store, request_c);
    Require(credit_c.result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED,
            "credit is granted under the replacement epoch");
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    const auto reacquired = Acquire(store, request_c);
    Require(reacquired.result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED &&
                reacquired.credit_id() != credit_c.credit_id(),
            "an expired reservation can be reacquired idempotently");

    auto conflicting_request = request_c;
    conflicting_request.set_batch_id("another-batch");
    Require(Acquire(store, conflicting_request).result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_REJECTED_INVALID,
            "a request_id cannot authorize another immutable batch");

    auto stale = MakeBatch("stale-credit", 1, 4, 0, 130);
    Require(Acquire(store, MakeCreditRequest(stale)).result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_REJECTED_FRESHNESS,
            "policy freshness rejects stale producer samples");

    auto old_contract = MakeCreditRequest(batch_c, "old-contract");
    old_contract.mutable_contract()->set_package_version("0.9.1");
    Require(Acquire(store, old_contract).result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_REJECTED_IDENTITY,
            "the previous contract version fails closed");

    rl::training::v1::ReleaseSampleDemandReq release_demand;
    *release_demand.mutable_consumer() = epoch_two.demand().consumer();
    *release_demand.mutable_contract() = epoch_two.demand().contract();
    release_demand.set_demand_id(epoch_two.demand().demand_id());
    release_demand.set_demand_epoch(epoch_two.demand().demand_epoch());
    rl::training::v1::SampleDemandRsp demand_release_response;
    store.ReleaseDemand(release_demand, &demand_release_response);
    Require(demand_release_response.result() ==
                rl::training::v1::SAMPLE_DEMAND_RESULT_RELEASED &&
                demand_release_response.reserved_samples() == 0 &&
                demand_release_response.reserved_fragments() == 0 &&
                Acquire(store, MakeCreditRequest(
                                   MakeBatch("draining", 1, 5, 2, 131)))
                        .result() ==
                    rl::training::v1::SAMPLE_CREDIT_RESULT_WAIT_DRAINING,
            "release drains the demand and leaves no reservation");
}

void TestPhysicalCapacityWaitIsDistinct() {
    DistributorConfig config = TestConfig();
    config.max_queue_samples = 128;
    SampleStore store(config);
    Require(Upsert(store, MakeDemand(1, 1024, 10)).result() ==
                rl::training::v1::SAMPLE_DEMAND_RESULT_APPLIED,
            "wide demand is applied");
    auto first = MakeBatch("physical-a", 100, 1);
    Require(Acquire(store, MakeCreditRequest(first)).result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED,
            "first physical reservation is granted");
    auto second = MakeBatch("physical-b", 64, 2, 0, 100);
    Require(Acquire(store, MakeCreditRequest(second)).result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_WAIT_CAPACITY,
            "physical capacity wait is distinct from demand-window wait");
}

void TestCancelledGetCannotCreateHiddenLease() {
    SampleStore store(TestConfig());
    const auto batch = MakeBatch("cancel-late-arrival", 1, 1);
    Require(Upsert(store, MakeDemand()).result() ==
                rl::training::v1::SAMPLE_DEMAND_RESULT_APPLIED,
            "cancellation test demand is applied");
    const auto credit = Acquire(store, MakeCreditRequest(batch));
    Require(credit.result() ==
                rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED,
            "late-arrival batch owns capacity before GetBatch waits");

    const auto request = MakeGetBatchRequest(1, 1000, 100, 0);
    rl::training::v1::GetBatchRsp cancelled_response;
    std::atomic<bool> cancelled{false};
    std::atomic<int> cancellation_checks{0};
    std::thread consumer([&]() {
        store.GetBatch(request, &cancelled_response, [&]() {
            cancellation_checks.fetch_add(1, std::memory_order_relaxed);
            return cancelled.load(std::memory_order_acquire);
        });
    });

    for (int attempt = 0;
         attempt < 500 &&
         cancellation_checks.load(std::memory_order_relaxed) < 2;
         ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Require(cancellation_checks.load(std::memory_order_relaxed) >= 2,
            "GetBatch reached its cancellation-aware wait");
    cancelled.store(true, std::memory_order_release);
    rl::training::v1::PushSamplesReq push_request;
    push_request.set_credit_id(credit.credit_id());
    *push_request.mutable_batch() = batch;
    rl::training::v1::PushSamplesRsp push_response;
    store.Push(push_request, &push_response);
    Require(push_response.result() == rl::training::v1::PUSH_RESULT_ACCEPTED,
            "sample arrives in the wake-up that observes cancellation");
    consumer.join();

    Require(cancelled_response.result() ==
                rl::training::v1::GET_BATCH_RESULT_REJECTED &&
                cancelled_response.delivery_id().empty(),
            "cancelled GetBatch returns without a delivery identity");
    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.leased_samples() == 0 &&
                status.ready_queue_samples() == 1,
            "cancelled late arrival remains READY with no hidden lease");

    const auto recovered = Get(store, 1, 100, 100, 0);
    Require(recovered.result() ==
                rl::training::v1::GET_BATCH_RESULT_LEASED &&
                recovered.batches(0).batch_id() == batch.batch_id(),
            "next request leases the exact late-arrival payload");
    Ack(store, recovered.delivery_id(),
        rl::training::v1::ACK_DISPOSITION_SHUTDOWN_UNTRAINED);
}

void TestCancellationDuringAssemblyRollsBackExactly() {
    SampleStore store(TestConfig());
    const auto batch = MakeBatch("cancel-before-commit", 12, 1);
    Require(Push(store, batch).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "pre-commit cancellation batch is ready");

    const auto request = MakeGetBatchRequest(12, 100, 100, 0);
    int cancellation_checks = 0;
    rl::training::v1::GetBatchRsp cancelled_response;
    store.GetBatch(request, &cancelled_response, [&]() {
        return ++cancellation_checks >= 3;
    });
    Require(cancelled_response.result() ==
                rl::training::v1::GET_BATCH_RESULT_REJECTED &&
                cancelled_response.delivery_id().empty(),
            "cancellation after assembly rejects before lease commit");

    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.ready_queue_samples() == 12 &&
                status.ready_queue_fragments() == 1 &&
                status.leased_samples() == 0 &&
                status.target_hit_count() == 0,
            "pre-commit cancellation restores READY accounting exactly");
    const auto recovered = Get(store, 12, 100, 100, 0);
    Require(recovered.result() ==
                rl::training::v1::GET_BATCH_RESULT_LEASED &&
                recovered.batches(0).batch_id() == batch.batch_id(),
            "rollback preserves FIFO identity for the next request");
    Ack(store, recovered.delivery_id(),
        rl::training::v1::ACK_DISPOSITION_SHUTDOWN_UNTRAINED);
}

void TestAlreadyExpiredHandlerCannotLeaseReadySamples() {
    SampleStore store(TestConfig());
    Require(Push(store, MakeBatch("expired-before-lock", 1, 1)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "already-expired handler sample is ready");
    const auto request = MakeGetBatchRequest(1, 100, 100, 0);
    rl::training::v1::GetBatchRsp response;
    store.GetBatch(request, &response, []() { return true; });
    Require(response.result() ==
                rl::training::v1::GET_BATCH_RESULT_REJECTED &&
                response.delivery_id().empty(),
            "first post-lock cancellation fence rejects the handler");
    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.ready_queue_samples() == 1 &&
                status.leased_samples() == 0,
            "a handler expired before mutex acquisition creates no lease");
}

void TestExpiredRequestCannotLeaseReadySamples() {
    SampleStore store(TestConfig());
    Require(Push(store, MakeBatch("deadline-before-lease", 1, 1)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "deadline test sample is ready");
    const auto request = MakeGetBatchRequest(1, 5, 100, 0);
    int cancellation_checks = 0;
    rl::training::v1::GetBatchRsp timed_out;
    store.GetBatch(request, &timed_out, [&]() {
        if (++cancellation_checks == 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
        }
        return false;
    });
    Require(timed_out.result() ==
                rl::training::v1::GET_BATCH_RESULT_TIMEOUT &&
                timed_out.delivery_id().empty(),
            "elapsed request deadline prevents a ready-sample lease");
    rl::training::v1::DistributorStatusRsp status;
    store.GetStatus({}, &status);
    Require(status.ready_queue_samples() == 1 &&
                status.leased_samples() == 0 &&
                status.empty_timeout_count() == 1,
            "deadline fence leaves no hidden lease and records one timeout");
}

void TestStatusLatencyAtObservedProductionScale() {
    DistributorConfig config = TestConfig();
    config.max_queue_samples = 12000;
    config.max_queue_fragments = 12000;
    config.max_dedup_entries = 30000;
    config.credit_ttl_ms = 60000;
    SampleStore store(config);
    Require(Upsert(store, MakeDemand(1, 12000, 12000, 1751, 1751, 60000))
                .result() ==
                rl::training::v1::SAMPLE_DEMAND_RESULT_APPLIED,
            "scale-test demand is applied");

    constexpr int kTerminalCreditCount = 20000;
    const auto terminal_batch = MakeBatch("status-terminal", 1, 1);
    for (int index = 0; index < kTerminalCreditCount; ++index) {
        const auto credit = Acquire(
            store,
            MakeCreditRequest(terminal_batch,
                              "status-terminal-request-" +
                                  std::to_string(index)));
        Require(credit.result() ==
                    rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED,
                "scale-test credit is granted");
        Require(ReleaseCredit(store, credit, terminal_batch).state() ==
                    rl::training::v1::SAMPLE_CREDIT_STATE_RELEASED,
                "scale-test credit becomes terminal");
    }

    constexpr int kBehaviorVersionCount = 1752;
    for (int version = 0; version < kBehaviorVersionCount; ++version) {
        const auto batch = MakeBatch(
            "status-policy-" + std::to_string(version), 1, version + 2,
            static_cast<uint64_t>(version),
            static_cast<uint64_t>(version));
        const auto credit = Acquire(store, MakeCreditRequest(batch));
        Require(credit.result() ==
                    rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED,
                "behavior-version credit is granted");
        rl::training::v1::PushSamplesReq request;
        request.set_credit_id(credit.credit_id());
        *request.mutable_batch() = batch;
        rl::training::v1::PushSamplesRsp response;
        store.Push(request, &response);
        Require(response.result() == rl::training::v1::PUSH_RESULT_ACCEPTED,
                "behavior-version batch is accepted");
    }

    constexpr int kAdditionalReadyBatchCount = 8192;
    for (int index = 0; index < kAdditionalReadyBatchCount; ++index) {
        const auto batch = MakeBatch(
            "status-ready-" + std::to_string(index), 1,
            kBehaviorVersionCount + index + 2, 1751,
            kBehaviorVersionCount + index);
        const auto credit = Acquire(store, MakeCreditRequest(batch));
        Require(credit.result() ==
                    rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED,
                "large READY-set credit is granted");
        rl::training::v1::PushSamplesReq request;
        request.set_credit_id(credit.credit_id());
        *request.mutable_batch() = batch;
        rl::training::v1::PushSamplesRsp response;
        store.Push(request, &response);
        Require(response.result() == rl::training::v1::PUSH_RESULT_ACCEPTED,
                "large READY-set batch is accepted");
    }

    constexpr int kStatusReads = 25;
    rl::training::v1::DistributorStatusRsp status;
    const auto started_at = std::chrono::steady_clock::now();
    for (int index = 0; index < kStatusReads; ++index) {
        store.GetStatus({}, &status);
    }
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started_at)
            .count();
    Require(status.credit_release_count() == kTerminalCreditCount &&
                status.reserved_samples() == 0 &&
                status.behavior_versions_size() == kBehaviorVersionCount &&
                status.ready_queue_samples() ==
                    kBehaviorVersionCount + kAdditionalReadyBatchCount,
            "large status snapshot preserves exact accounting");
    Require(elapsed_ms < 2000,
            "25 status snapshots at observed scale complete below 2s, got " +
                std::to_string(elapsed_ms) + "ms");

    std::atomic<bool> stop_status_polling{false};
    std::atomic<int64_t> concurrent_status_reads{0};
    std::thread status_poller([&]() {
        rl::training::v1::DistributorStatusRsp concurrent_status;
        while (!stop_status_polling.load(std::memory_order_acquire)) {
            store.GetStatus({}, &concurrent_status);
            concurrent_status_reads.fetch_add(1, std::memory_order_release);
        }
    });
    for (int attempt = 0;
         attempt < 500 &&
         concurrent_status_reads.load(std::memory_order_acquire) < 5;
         ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Require(concurrent_status_reads.load(std::memory_order_acquire) >= 5,
            "status poller is active before data-plane latency measurement");

    constexpr int kDataPlaneCycles = 128;
    int64_t maximum_get_us = 0;
    int64_t maximum_ack_us = 0;
    int64_t maximum_credit_us = 0;
    int64_t maximum_push_us = 0;
    auto get_request = MakeGetBatchRequest(1, 200, 1000, 1751);
    get_request.mutable_freshness()->set_max_version_lag(1751);
    for (int index = 0; index < kDataPlaneCycles; ++index) {
        rl::training::v1::GetBatchRsp delivery;
        const auto get_started_at = std::chrono::steady_clock::now();
        store.GetBatch(get_request, &delivery, []() { return false; });
        maximum_get_us = std::max(
            maximum_get_us,
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - get_started_at)
                .count());
        Require(delivery.result() ==
                    rl::training::v1::GET_BATCH_RESULT_LEASED,
                "GetBatch meets its 200ms deadline during status polling");

        const auto ack_started_at = std::chrono::steady_clock::now();
        const auto ack = Ack(
            store, delivery.delivery_id(),
            rl::training::v1::ACK_DISPOSITION_SHUTDOWN_UNTRAINED);
        maximum_ack_us = std::max(
            maximum_ack_us,
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - ack_started_at)
                .count());
        Require(ack.result() == rl::training::v1::DELIVERY_RESULT_APPLIED,
                "Ack completes during status polling");

        const auto replacement = MakeBatch(
            "status-replacement-" + std::to_string(index), 1,
            kBehaviorVersionCount + kAdditionalReadyBatchCount + index + 2,
            1751,
            kBehaviorVersionCount + kAdditionalReadyBatchCount + index);
        const auto credit_started_at = std::chrono::steady_clock::now();
        const auto credit = Acquire(store, MakeCreditRequest(replacement));
        maximum_credit_us = std::max(
            maximum_credit_us,
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - credit_started_at)
                .count());
        Require(credit.result() ==
                    rl::training::v1::SAMPLE_CREDIT_RESULT_GRANTED,
                "AcquireCredit completes during status polling");

        rl::training::v1::PushSamplesReq push_request;
        push_request.set_credit_id(credit.credit_id());
        *push_request.mutable_batch() = replacement;
        rl::training::v1::PushSamplesRsp push_response;
        const auto push_started_at = std::chrono::steady_clock::now();
        store.Push(push_request, &push_response);
        maximum_push_us = std::max(
            maximum_push_us,
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - push_started_at)
                .count());
        Require(push_response.result() ==
                    rl::training::v1::PUSH_RESULT_ACCEPTED,
                "Push completes during status polling");
    }
    stop_status_polling.store(true, std::memory_order_release);
    status_poller.join();

    constexpr int64_t kMaximumDataPlaneLatencyUs = 200000;
    const int64_t status_reads_during_data_plane =
        concurrent_status_reads.load(std::memory_order_acquire);
    Require(status_reads_during_data_plane >= 5,
            "data-plane operations overlap status projections");
    Require(maximum_get_us < kMaximumDataPlaneLatencyUs &&
                maximum_ack_us < kMaximumDataPlaneLatencyUs &&
                maximum_credit_us < kMaximumDataPlaneLatencyUs &&
                maximum_push_us < kMaximumDataPlaneLatencyUs,
            "status polling keeps every data-plane operation below 200ms: "
            "get=" +
                std::to_string(maximum_get_us) + "us ack=" +
                std::to_string(maximum_ack_us) + "us credit=" +
                std::to_string(maximum_credit_us) + "us push=" +
                std::to_string(maximum_push_us) + "us");
    std::cout << "status_concurrency: reads="
              << status_reads_during_data_plane
              << " max_get_us=" << maximum_get_us
              << " max_ack_us=" << maximum_ack_us
              << " max_credit_us=" << maximum_credit_us
              << " max_push_us=" << maximum_push_us << std::endl;
}

}  // namespace

int main() {
    TestPushIdentityAndValidation();
    TestTerminalAndBootstrapContract();
    TestBoundedCompletedDedupHistory();
    TestBoundedMultiVersionAssemblyAndAck();
    TestFreshnessRejectsOldButRetainsFuture();
    TestMultipleProducersAndDrain();
    TestRenewExpiryAndSingleConsumer();
    TestDemandCreditFlowControl();
    TestPhysicalCapacityWaitIsDistinct();
    TestCancelledGetCannotCreateHiddenLease();
    TestCancellationDuringAssemblyRollsBackExactly();
    TestAlreadyExpiredHandlerCannotLeaseReadySamples();
    TestExpiredRequestCannotLeaseReadySamples();
    TestStatusLatencyAtObservedProductionScale();
    std::cout << "sample_store_contract: PASS" << std::endl;
    return 0;
}
