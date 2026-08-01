#include "store/sample_store.h"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

namespace {

int Fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

void Require(bool condition, const std::string& message) {
    if (!condition) std::exit(Fail(message));
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
    return config;
}

maze::SampleBatch MakeBatch(const std::string& batch_id,
                            int samples,
                            uint64_t sequence,
                            int version = 0,
                            int64_t first_frame = 0,
                            const std::string& producer = "producer-0") {
    maze::SampleBatch batch;
    batch.set_aiserver_id("aiserver-0");
    batch.set_env_id("env-0");
    batch.set_session_id(0);
    batch.set_episode_id(0);
    batch.set_agent_id(static_cast<int>(sequence % 4));
    batch.set_fragment_id(static_cast<int>(sequence));
    batch.set_behavior_model_version(version);
    batch.set_behavior_model_checksum(std::string(64, 'a' + version));
    batch.set_created_ts_ms(1);
    batch.set_producer_instance_id(producer);
    batch.set_fragment_seq(sequence);
    batch.set_batch_id(batch_id);
    batch.set_protocol_version(3);
    batch.set_termination_reason(maze::TERMINATION_REASON_ACTIVE);
    batch.set_bootstrap_value(0.25f);
    batch.set_bootstrap_valid(true);
    batch.set_first_action_frame_id(first_frame);
    batch.set_last_action_frame_id(first_frame + samples - 1);
    for (int index = 0; index < samples; ++index) {
        auto* sample = batch.add_samples();
        sample->add_obs(static_cast<float>(index));
        sample->set_action(index % 9);
        sample->set_reward(0.1f);
        sample->set_old_log_prob(-0.5f);
        sample->set_old_vpred(0.2f);
        sample->set_termination_reason(maze::TERMINATION_REASON_ACTIVE);
        sample->set_action_frame_id(first_frame + index);
    }
    return batch;
}

maze::PushSamplesRsp Push(SampleStore& store,
                          const maze::SampleBatch& batch) {
    maze::PushSamplesRsp response;
    store.Push(batch, &response);
    return response;
}

maze::GetBatchRsp Get(
    SampleStore& store,
    int target,
    int timeout_ms,
    int lease_timeout_ms,
    int version,
    maze::BatchSelectionPolicy policy =
        maze::BATCH_SELECTION_POLICY_TARGET_ONLY,
    const std::string& consumer = "consumer-0") {
    maze::GetBatchReq request;
    request.set_consumer_instance_id(consumer);
    request.set_batch_size(target);
    request.set_timeout_ms(timeout_ms);
    request.set_lease_timeout_ms(lease_timeout_ms);
    request.set_behavior_model_version(version);
    request.set_selection_policy(policy);
    maze::GetBatchRsp response;
    store.GetBatch(request, &response, []() { return false; });
    return response;
}

maze::DeliveryRsp Ack(
    SampleStore& store,
    const std::string& delivery_id,
    maze::AckDisposition disposition,
    const std::string& train_update_id = "",
    const std::string& consumer = "consumer-0") {
    maze::AckBatchReq request;
    request.set_consumer_instance_id(consumer);
    request.set_delivery_id(delivery_id);
    request.set_disposition(disposition);
    request.set_train_update_id(train_update_id);
    maze::DeliveryRsp response;
    store.Ack(request, &response);
    return response;
}

void TestPushIdentityAndValidation() {
    DistributorConfig config = TestConfig();
    config.max_queue_samples = 2;
    SampleStore store(config);
    auto first = MakeBatch("batch-1", 2, 1);
    Require(Push(store, first).result() == maze::PUSH_RESULT_ACCEPTED,
            "first batch must be accepted");
    Require(Push(store, first).result() == maze::PUSH_RESULT_DUPLICATE,
            "same batch_id must be idempotent");
    Require(
        Push(store, MakeBatch("batch-2", 1, 2)).result() ==
            maze::PUSH_RESULT_REJECTED_CAPACITY,
        "sample capacity must reject a new unique batch");

    auto invalid = MakeBatch("batch-invalid", 1, 3);
    invalid.set_bootstrap_valid(false);
    Require(Push(store, invalid).result() ==
                maze::PUSH_RESULT_REJECTED_INVALID,
            "fragment without bootstrap must be rejected");

    maze::DistributorStatusRsp status;
    store.GetStatus(maze::DistributorStatusReq{}, &status);
    Require(status.accepted_unique_samples() == 2,
            "accepted count excludes duplicate and rejected attempts");
    Require(status.duplicate_push_attempt_count() == 1,
            "duplicate attempt count");
    Require(status.rejected_push_attempt_count() == 2,
            "rejected attempt count");
}

void TestExactVersionTargetOnlyAndAck() {
    SampleStore store(TestConfig());
    Push(store, MakeBatch("v0-a", 60, 1, 0));
    Push(store, MakeBatch("v1-a", 512, 2, 1));
    Push(store, MakeBatch("v0-b", 60, 3, 0, 60));

    auto v0 = Get(store, 100, 20, 100, 0);
    Require(v0.result() == maze::GET_BATCH_RESULT_LEASED,
            "matching version target must lease");
    Require(v0.actual_batch_size() == 120,
            "whole fragments may overshoot target");
    Require(v0.batches_size() == 2 &&
                v0.batches(0).batch_id() == "v0-a" &&
                v0.batches(1).batch_id() == "v0-b",
            "version FIFO must ignore interleaved other versions");
    auto ack = Ack(
        store, v0.delivery_id(), maze::ACK_DISPOSITION_TRAINED,
        "update-v0");
    Require(ack.result() == maze::DELIVERY_RESULT_APPLIED,
            "trained Ack applies");
    auto duplicate = Ack(
        store, v0.delivery_id(), maze::ACK_DISPOSITION_TRAINED,
        "update-v0");
    Require(duplicate.result() == maze::DELIVERY_RESULT_ALREADY_APPLIED,
            "trained Ack retry is idempotent");
    auto conflict = Ack(
        store, v0.delivery_id(), maze::ACK_DISPOSITION_STALE);
    Require(conflict.result() == maze::DELIVERY_RESULT_REJECTED,
            "Ack retry cannot change disposition");

    auto v1 = Get(store, 512, 20, 100, 1);
    Require(v1.actual_batch_size() == 512 &&
                v1.behavior_model_version() == 1,
            "exact behavior version must be returned");
    Ack(
        store, v1.delivery_id(), maze::ACK_DISPOSITION_TRAINED,
        "update-v1");

    maze::DistributorStatusRsp status;
    store.GetStatus(maze::DistributorStatusReq{}, &status);
    Require(status.trained_sample_count() == 632,
            "trained samples are counted by disposition");
    Require(status.behavior_versions_size() == 2,
            "status exposes both behavior versions");
}

void TestMultipleProducersShareVersionFifo() {
    SampleStore store(TestConfig());
    auto first = MakeBatch("producer-0-fragment-1", 64, 1);
    auto second = MakeBatch(
        "producer-1-fragment-1", 64, 1, 0, 0, "producer-1");

    Require(Push(store, first).result() == maze::PUSH_RESULT_ACCEPTED,
            "first producer batch is accepted");
    Require(Push(store, second).result() == maze::PUSH_RESULT_ACCEPTED,
            "second producer batch is accepted");
    auto delivery = Get(store, 128, 20, 100, 0);
    Require(delivery.result() == maze::GET_BATCH_RESULT_LEASED,
            "batches from both producers are leased together");
    Require(delivery.batches_size() == 2 &&
                delivery.batches(0).producer_instance_id() == "producer-0" &&
                delivery.batches(1).producer_instance_id() == "producer-1",
            "producer identity and insertion order remain distinct");
    Ack(
        store, delivery.delivery_id(),
        maze::ACK_DISPOSITION_SHUTDOWN_UNTRAINED);
}

void TestTargetOnlyDoesNotReturnPartial() {
    SampleStore store(TestConfig());
    Push(store, MakeBatch("partial", 50, 1, 0));
    auto timeout = Get(store, 512, 10, 100, 0);
    Require(timeout.result() == maze::GET_BATCH_RESULT_TIMEOUT,
            "TARGET_ONLY must not lease a partial batch");

    auto drain = Get(
        store, 512, 10, 100, 0,
        maze::BATCH_SELECTION_POLICY_DRAIN_AVAILABLE);
    Require(drain.result() == maze::GET_BATCH_RESULT_LEASED &&
                drain.actual_batch_size() == 50,
            "DRAIN_AVAILABLE may lease a partial batch");

    maze::NackBatchReq nack;
    nack.set_consumer_instance_id("consumer-0");
    nack.set_delivery_id(drain.delivery_id());
    nack.set_reason("contract test");
    maze::DeliveryRsp response;
    store.Nack(nack, &response);
    Require(response.result() == maze::DELIVERY_RESULT_APPLIED,
            "Nack requeues a drain delivery");
    auto redelivery = Get(
        store, 512, 10, 100, 0,
        maze::BATCH_SELECTION_POLICY_DRAIN_AVAILABLE);
    Require(redelivery.batches(0).batch_id() == "partial",
            "Nack preserves per-version FIFO order");
    Ack(
        store, redelivery.delivery_id(),
        maze::ACK_DISPOSITION_SHUTDOWN_UNTRAINED);
}

void TestRenewAndExpiry() {
    SampleStore store(TestConfig());
    Push(store, MakeBatch("renew", 12, 1, 0));
    auto first = Get(store, 12, 10, 20, 0);
    Require(first.result() == maze::GET_BATCH_RESULT_LEASED,
            "initial lease");

    maze::RenewLeaseReq renew;
    renew.set_consumer_instance_id("consumer-0");
    renew.set_delivery_id(first.delivery_id());
    renew.set_lease_timeout_ms(100);
    maze::DeliveryRsp renew_response;
    store.RenewLease(renew, &renew_response);
    Require(renew_response.result() == maze::DELIVERY_RESULT_APPLIED,
            "owner may renew a lease");
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    Require(
        Ack(
            store, first.delivery_id(), maze::ACK_DISPOSITION_STALE)
                .result() == maze::DELIVERY_RESULT_APPLIED,
        "renewed lease remains valid");

    Push(store, MakeBatch("expire", 12, 2, 0, 20));
    auto expiring = Get(store, 12, 10, 20, 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    auto redelivery = Get(store, 12, 10, 100, 0);
    Require(redelivery.result() == maze::GET_BATCH_RESULT_LEASED &&
                redelivery.delivery_id() != expiring.delivery_id() &&
                redelivery.batches(0).batch_id() == "expire",
            "expired lease is redelivered with original batch identity");
    Ack(
        store, redelivery.delivery_id(), maze::ACK_DISPOSITION_INVALID);

    maze::DistributorStatusRsp status;
    store.GetStatus(maze::DistributorStatusReq{}, &status);
    Require(status.lease_renew_count() == 1,
            "lease renewal count");
    Require(status.expired_lease_count() == 1,
            "expired lease count");
    Require(status.stale_sample_count() == 12 &&
                status.invalid_sample_count() == 12,
            "non-training Ack dispositions remain distinct");
}

void TestSingleConsumerCapability() {
    SampleStore store(TestConfig());
    Push(store, MakeBatch("single-consumer", 12, 1, 0));
    auto first = Get(store, 12, 10, 100, 0);
    Require(first.result() == maze::GET_BATCH_RESULT_LEASED,
            "first consumer leases the batch");

    auto second = Get(
        store, 12, 10, 100, 0,
        maze::BATCH_SELECTION_POLICY_TARGET_ONLY, "consumer-1");
    Require(second.result() == maze::GET_BATCH_RESULT_BUSY,
            "second consumer is rejected while a lease is active");

    maze::DistributorStatusRsp status;
    store.GetStatus(maze::DistributorStatusReq{}, &status);
    Require(
        status.backend_type() == maze::SAMPLE_BACKEND_TYPE_LOCAL_MEMORY,
        "status identifies the local-memory backend");
    Require(status.max_concurrent_consumers() == 1,
            "local backend declares one consumer");
    Require(status.active_consumer_count() == 1,
            "active lease is reported as one consumer");
    Require(status.consumer_busy_count() == 1,
            "busy consumer attempts are counted");
    Require(status.ingress_ready() && status.pool_ready(),
            "combined local service exposes both readiness states");

    Ack(
        store, first.delivery_id(),
        maze::ACK_DISPOSITION_SHUTDOWN_UNTRAINED);
}

}  // namespace

int main() {
    TestPushIdentityAndValidation();
    TestExactVersionTargetOnlyAndAck();
    TestMultipleProducersShareVersionFifo();
    TestTargetOnlyDoesNotReturnPartial();
    TestRenewAndExpiry();
    TestSingleConsumerCapability();
    std::cout << "sample_store_contract: PASS" << std::endl;
    return 0;
}
