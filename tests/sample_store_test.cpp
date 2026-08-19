#include "store/sample_pool_coordinator.h"
#include "store/sample_store_backend.h"

#include <chrono>
#include <cstdlib>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <iomanip>
#include <iostream>
#include <openssl/evp.h>
#include <sstream>
#include <string>

namespace {

constexpr char kSourceDigest[] =
    "01986363f0cf21b3eeaa48a84f1fd6858fc2a8b2c76dfa62ba0e2790e283f936";
constexpr char kArtifactDigest[] =
    "00de3bd57978ad781a82dbb839f98414751f2f6960358ab4e0c43a7c51d845dd";
constexpr char kGeneratorIdentity[] =
    "0eb73fc2cb675bdb34bf3db9c99dae62a82f93a5e3a72db84dcf3936464729c8";

int Fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

void Require(bool condition, const std::string& message) {
    if (!condition) std::exit(Fail(message));
}

int64_t NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
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
    contract->set_package_version("0.13.0");
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

SamplePoolConfig TestConfig() {
    SamplePoolConfig config;
    config.max_queue_samples = 10000;
    config.max_queue_fragments = 100;
    config.max_fragment_samples = 128;
    config.max_queue_estimated_bytes = 64 * 1024 * 1024;
    config.max_dedup_entries = 1000;
    config.delivery_history_size = 1000;
    config.default_get_timeout_ms = 10;
    config.default_lease_timeout_ms = 100;
    config.contract.package_name = "rl-contracts";
    config.contract.package_version = "0.13.0";
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
    uint64_t model_step = 0,
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
    batch.mutable_behavior_policy()->set_model_lineage_id("lineage-a");
    batch.mutable_behavior_policy()->set_model_step(model_step);
    batch.mutable_behavior_policy()->set_distribution_schema_id(
        "categorical.logits.v1");
    SetDigest(batch.mutable_behavior_policy()->mutable_policy_spec_digest(),
              std::string(64, '5'));
    *batch.mutable_training_semantics() = MakeSemantics();
    FillService(batch.mutable_producer(), "aiserver", "producer-0");
    FillContract(batch.mutable_contract());
    batch.set_created_at_unix_ms(created_at_unix_ms > 0
                                     ? created_at_unix_ms
                                     : NowMs());
    batch.set_first_action_step(sequence * 1000);
    batch.set_last_action_step(sequence * 1000 + sample_count - 1);
    for (int index = 0; index < sample_count; ++index) {
        auto* sample = batch.add_samples();
        sample->add_observation(static_cast<float>(index));
        sample->add_observation(0.25f);
        sample->add_next_observation(static_cast<float>(index + 1));
        sample->add_next_observation(0.5f);
        sample->set_action(index % 9);
        sample->set_reward(0.0f);
        sample->set_old_log_probability(-0.5f);
        sample->set_old_value_prediction(0.2f);
        sample->set_end_kind(
            rl::training::v1::TRANSITION_END_KIND_CONTINUING);
        sample->set_action_step(sequence * 1000 + index);
    }
    RefreshPayloadDigest(&batch);
    return batch;
}

rl::training::v1::PushSamplesRsp Push(
    SamplePoolCoordinator& pool,
    const rl::training::v1::SampleBatch& batch) {
    rl::training::v1::PushSamplesReq request;
    *request.mutable_batch() = batch;
    rl::training::v1::PushSamplesRsp response;
    pool.Push(request, &response);
    return response;
}

rl::training::v1::GetBatchReq MakeGetRequest(
    int target_samples,
    uint64_t reference_model_step,
    uint32_t max_model_step_lag = 1000,
    int64_t max_sample_age_ms = 60000,
    rl::training::v1::BatchAssemblyMode mode =
        rl::training::v1::BATCH_ASSEMBLY_MODE_TARGET_BOUNDED) {
    rl::training::v1::GetBatchReq request;
    request.mutable_assembly()->set_target_samples(target_samples);
    request.mutable_assembly()->set_max_samples(target_samples + 127);
    request.mutable_assembly()->set_mode(mode);
    request.set_timeout_ms(10);
    request.set_lease_timeout_ms(1000);
    FillService(request.mutable_consumer(), "learner", "consumer-0");
    request.mutable_freshness()->set_model_lineage_id("lineage-a");
    request.mutable_freshness()->set_reference_model_step(
        reference_model_step);
    request.mutable_freshness()->set_max_model_step_lag(max_model_step_lag);
    request.mutable_freshness()->set_max_sample_age_ms(max_sample_age_ms);
    request.mutable_freshness()->set_distribution_schema_id(
        "categorical.logits.v1");
    SetDigest(request.mutable_freshness()->mutable_policy_spec_digest(),
              std::string(64, '5'));
    *request.mutable_required_semantics() = MakeSemantics();
    return request;
}

rl::training::v1::GetBatchRsp Get(
    SamplePoolCoordinator& pool,
    const rl::training::v1::GetBatchReq& request) {
    rl::training::v1::GetBatchRsp response;
    pool.GetBatch(request, &response, []() { return false; });
    return response;
}

rl::training::v1::DeliveryRsp Ack(
    SamplePoolCoordinator& pool,
    const rl::training::v1::GetBatchRsp& delivery,
    rl::training::v1::AckDisposition disposition,
    const std::string& train_update_id = "") {
    rl::training::v1::AckBatchReq request;
    FillService(request.mutable_consumer(), "learner", "consumer-0");
    request.set_delivery_id(delivery.delivery_id());
    request.set_disposition(disposition);
    request.set_train_update_id(train_update_id);
    rl::training::v1::DeliveryRsp response;
    pool.Ack(request, &response);
    return response;
}

rl::training::v1::SamplePoolStatusRsp Status(
    SamplePoolCoordinator& pool) {
    rl::training::v1::SamplePoolStatusRsp response;
    pool.GetStatus({}, &response);
    return response;
}

rl::training::v1::FinalizeSamplePoolRsp Finalize(
    SamplePoolCoordinator& pool,
    const std::string& finalization_id,
    const std::string& consumer_instance = "consumer-0") {
    const auto status = Status(pool);
    rl::training::v1::FinalizeSamplePoolReq request;
    FillService(request.mutable_consumer(), "learner", consumer_instance);
    *request.mutable_expected_sample_pool() = status.sample_pool();
    request.set_finalization_id(finalization_id);
    rl::training::v1::FinalizeSamplePoolRsp response;
    pool.FinalizeSamplePool(request, &response);
    return response;
}

void TestUidDigestAndDirectIngress() {
    SamplePoolCoordinator pool(TestConfig());
    const auto batch = MakeBatch("uid-1", 2, 1);
    const auto accepted = Push(pool, batch);
    Require(accepted.result() == rl::training::v1::PUSH_RESULT_ACCEPTED &&
                accepted.accepted_samples() == 2 &&
                accepted.sample_pool().component() == "sample-pool",
            "Push accepts a complete fragment without Demand or Credit");

    const auto duplicate = Push(pool, batch);
    Require(duplicate.result() == rl::training::v1::PUSH_RESULT_DUPLICATE,
            "same UID and digest is an idempotent duplicate");

    auto conflicting = batch;
    conflicting.mutable_samples(0)->set_action(8);
    RefreshPayloadDigest(&conflicting);
    const auto conflict = Push(pool, conflicting);
    Require(conflict.result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_CONFLICT,
            "same UID with different bytes is a conflict");

    const auto status = Status(pool);
    Require(status.backend_type() ==
                rl::training::v1::SAMPLE_BACKEND_TYPE_LOCAL_MEMORY &&
                status.push_attempt_count() == 3 &&
                status.accepted_unique_batches() == 1 &&
                status.duplicate_push_attempt_count() == 1 &&
                status.rejected_push_attempt_count() == 1,
            "status reports LocalFragmentStore ingress accounting");
}

void TestReadyFifoEviction() {
    auto config = TestConfig();
    config.max_queue_samples = 4;
    config.max_queue_fragments = 2;
    SamplePoolCoordinator pool(config);
    Require(Push(pool, MakeBatch("fifo-oldest", 2, 1)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "first FIFO fragment accepted");
    Require(Push(pool, MakeBatch("fifo-middle", 2, 2)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "second FIFO fragment accepted");
    Require(Push(pool, MakeBatch("fifo-newest", 2, 3)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "new fragment evicts the oldest READY fragment");

    const auto status = Status(pool);
    Require(status.ready_queue_samples() == 4 &&
                status.ready_queue_fragments() == 2 &&
                status.evicted_sample_count() == 2 &&
                status.evicted_fragment_count() == 1,
            "READY FIFO eviction preserves the configured hard bounds");

    const auto delivery = Get(pool, MakeGetRequest(2, 0));
    Require(delivery.result() ==
                rl::training::v1::GET_BATCH_RESULT_LEASED &&
                delivery.batches_size() == 1 &&
                delivery.batches(0).batch_id() == "fifo-middle",
            "global FIFO evicts oldest and leases the next READY fragment");
}

void TestLeasedProtectionAndOversizedReject() {
    auto config = TestConfig();
    config.max_queue_samples = 2;
    config.max_queue_fragments = 1;
    SamplePoolCoordinator pool(config);
    Require(Push(pool, MakeBatch("leased", 2, 1)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "fragment accepted before lease");
    const auto delivery = Get(pool, MakeGetRequest(2, 0));
    Require(delivery.result() == rl::training::v1::GET_BATCH_RESULT_LEASED,
            "fragment leased");
    Require(Push(pool, MakeBatch("cannot-evict-lease", 1, 2)).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_CAPACITY,
            "LEASED fragment is protected from capacity eviction");

    auto single_limit = TestConfig();
    single_limit.max_queue_samples = 1;
    SamplePoolCoordinator small_pool(single_limit);
    Require(Push(small_pool, MakeBatch("oversized", 2, 3)).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_CAPACITY,
            "a single fragment above the hard capacity is rejected");
}

void TestSelectionDoesNotDestructivelyExpire() {
    SamplePoolCoordinator pool(TestConfig());
    const int64_t old_time = NowMs() - 120000;
    Require(Push(pool, MakeBatch("old-step", 2, 1, 0, old_time)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "old provenance fragment accepted");
    Require(Push(pool, MakeBatch("current-step", 2, 2, 5)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "current provenance fragment accepted");

    const auto delivery = Get(pool, MakeGetRequest(2, 5, 0, 1000));
    Require(delivery.result() == rl::training::v1::GET_BATCH_RESULT_LEASED &&
                delivery.batches_size() == 1 &&
                delivery.batches(0).batch_id() == "current-step" &&
                delivery.minimum_behavior_model_step() == 5 &&
                delivery.maximum_behavior_model_step() == 5,
            "consumer selection leases one compatible behavior step");
    Require(Ack(pool, delivery, rl::training::v1::ACK_DISPOSITION_TRAINED,
                "update-1")
                .result() == rl::training::v1::DELIVERY_RESULT_APPLIED,
            "selected delivery can be trained and acknowledged");

    const auto status = Status(pool);
    Require(status.ready_queue_samples() == 2 &&
                status.ready_queue_fragments() == 1 &&
                status.stale_sample_count() == 0 &&
                status.evicted_sample_count() == 0,
            "unselected old step/age remains READY and is not hard-expired");
}

void TestWholeFragmentAssemblyAckNackRenewAndDrain() {
    SamplePoolCoordinator pool(TestConfig());
    Require(Push(pool, MakeBatch("assembly-a", 2, 1)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED &&
                Push(pool, MakeBatch("assembly-b", 2, 2)).result() ==
                    rl::training::v1::PUSH_RESULT_ACCEPTED,
            "whole fragments accepted for assembly");
    const auto delivery = Get(pool, MakeGetRequest(3, 0));
    Require(delivery.result() == rl::training::v1::GET_BATCH_RESULT_LEASED &&
                delivery.returned_samples() == 4 &&
                delivery.returned_fragments() == 2,
            "target assembly preserves whole fragment boundaries");

    rl::training::v1::RenewLeaseReq renew;
    FillService(renew.mutable_consumer(), "learner", "consumer-0");
    renew.set_delivery_id(delivery.delivery_id());
    renew.set_lease_timeout_ms(2000);
    rl::training::v1::DeliveryRsp renewed;
    pool.RenewLease(renew, &renewed);
    Require(renewed.result() == rl::training::v1::DELIVERY_RESULT_APPLIED,
            "active lease renews");

    rl::training::v1::NackBatchReq nack;
    FillService(nack.mutable_consumer(), "learner", "consumer-0");
    nack.set_delivery_id(delivery.delivery_id());
    nack.set_reason("retry delivery");
    rl::training::v1::DeliveryRsp nacked;
    pool.Nack(nack, &nacked);
    Require(nacked.result() == rl::training::v1::DELIVERY_RESULT_APPLIED &&
                Status(pool).ready_queue_samples() == 4,
            "Nack restores complete leased fragments");

    const auto redelivery = Get(pool, MakeGetRequest(3, 0));
    Require(redelivery.result() ==
                rl::training::v1::GET_BATCH_RESULT_LEASED &&
                Ack(pool, redelivery,
                    rl::training::v1::ACK_DISPOSITION_TRAINED,
                    "update-2")
                        .result() ==
                    rl::training::v1::DELIVERY_RESULT_APPLIED,
            "redelivery settles through Ack(TRAINED)");

    Require(Push(pool, MakeBatch("finalize-tail-a", 2, 3)).result() ==
                rl::training::v1::PUSH_RESULT_ACCEPTED,
            "tail fragment accepted");
    const auto active_tail = Get(pool, MakeGetRequest(2, 0));
    Require(active_tail.result() ==
                rl::training::v1::GET_BATCH_RESULT_LEASED,
            "tail fragment can be leased before finalization");
    const auto busy_finalize = Finalize(pool, "finalize-1");
    Require(busy_finalize.result() ==
                rl::training::v1::
                    SAMPLE_POOL_FINALIZE_RESULT_REJECTED_ACTIVE_LEASE &&
                busy_finalize.leased_samples() == 2 &&
                Status(pool).leased_samples() == 2,
            "active lease is preserved and blocks finalization");

    rl::training::v1::NackBatchReq tail_nack;
    FillService(tail_nack.mutable_consumer(), "learner", "consumer-0");
    tail_nack.set_delivery_id(active_tail.delivery_id());
    tail_nack.set_reason("settle lease before finalization");
    rl::training::v1::DeliveryRsp tail_nacked;
    pool.Nack(tail_nack, &tail_nacked);
    Require(tail_nacked.result() ==
                rl::training::v1::DELIVERY_RESULT_APPLIED &&
                Push(pool, MakeBatch("finalize-tail-b", 2, 4)).result() ==
                    rl::training::v1::PUSH_RESULT_ACCEPTED,
            "settled lease restores READY ownership before finalization");

    const auto finalized = Finalize(pool, "finalize-1");
    Require(finalized.result() ==
                rl::training::v1::SAMPLE_POOL_FINALIZE_RESULT_FINALIZED &&
                finalized.settled_samples() == 4 &&
                finalized.settled_fragments() == 2 &&
                finalized.ready_samples() == 0 &&
                finalized.leased_samples() == 0 &&
                finalized.resident_samples() == 0,
            "finalization atomically settles the complete READY tail");
    const auto repeated = Finalize(pool, "finalize-1");
    Require(repeated.result() ==
                rl::training::v1::
                    SAMPLE_POOL_FINALIZE_RESULT_ALREADY_FINALIZED &&
                repeated.settled_samples() == 4,
            "same finalization identity is idempotent");
    Require(Finalize(pool, "conflicting-finalize").result() ==
                rl::training::v1::
                    SAMPLE_POOL_FINALIZE_RESULT_REJECTED_CONFLICT,
            "a conflicting finalization identity fails closed");
    Require(Push(pool, MakeBatch("push-after-finalize", 1, 5)).result() ==
                rl::training::v1::PUSH_RESULT_REJECTED_FINALIZED,
            "finalized SamplePool rejects new ingress");

    const auto status = Status(pool);
    Require(status.resident_samples() == 0 &&
                status.ready_queue_samples() == 0 &&
                status.leased_samples() == 0 &&
                status.finalized() &&
                status.finalization_id() == "finalize-1" &&
                status.finalized_sample_count() == 4 &&
                status.finalized_fragment_count() == 2 &&
                status.shutdown_untrained_sample_count() == 4 &&
                status.acked_unique_samples() == 8 &&
                status.accepted_unique_samples() == 8 &&
                !status.ingress_ready() && !status.pool_ready() &&
                status.lease_renew_count() == 1 &&
                status.nack_count() == 2 &&
                status.redelivery_count() == 2,
            "delivery and finalization accounting converges exactly");
}

void TestBackendFifoSeamIsConcrete() {
    LocalFragmentStore backend;
    StoredFragment first;
    first.batch.set_batch_id("first");
    first.policy_key = "policy";
    first.sample_count = 1;
    first.batch.set_created_at_unix_ms(1);
    StoredFragment second;
    second.batch.set_batch_id("second");
    second.policy_key = "policy";
    second.sample_count = 1;
    second.batch.set_created_at_unix_ms(2);
    backend.PushBack(std::move(first));
    backend.PushBack(std::move(second));
    Require(std::string(backend.name()) == "LocalFragmentStore" &&
                backend.EvictOldestReady().batch.batch_id() == "first",
            "A3 backend seam is a concrete global FIFO LocalFragmentStore");
    const auto finalized_tail = backend.ExtractAllReady();
    Require(finalized_tail.size() == 1 &&
                finalized_tail.front().batch.batch_id() == "second" &&
                backend.ready().empty(),
            "backend seam atomically transfers READY ownership on finalize");
}

}  // namespace

int main() {
    TestUidDigestAndDirectIngress();
    TestReadyFifoEviction();
    TestLeasedProtectionAndOversizedReject();
    TestSelectionDoesNotDestructivelyExpire();
    TestWholeFragmentAssemblyAckNackRenewAndDrain();
    TestBackendFifoSeamIsConcrete();
    std::cout << "sample_store_contract: PASS" << std::endl;
    return 0;
}
