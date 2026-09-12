#include "store/sample_pool_coordinator.h"
#include "grpc/sample_pool_service.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <future>
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

namespace {

void Require(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << std::endl;
    std::exit(1);
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
    return config;
}

void FillBehaviorModel(rl::training::v1::ModelIdentity* model) {
    model->set_model_lineage_id("lineage-fixed");
    model->set_model_step(3);
}

rl::training::v1::ProcessedTransitionEnvelope MakeEnvelope() {
    rl::training::v1::ProcessedTransitionEnvelope envelope;
    envelope.set_envelope_id("envelope-fixed");
    FillService(envelope.mutable_producer(),
                "aiserver", "aiserver-fixed");
    FillBehaviorModel(envelope.mutable_behavior_model());

    for (uint32_t index = 0; index < 2; ++index) {
        auto* transition = envelope.add_samples();
        transition->set_item_id("item-" + std::to_string(index));
        transition->add_observation(static_cast<float>(index));
        transition->add_observation(0.25f);
        transition->set_action(static_cast<int32_t>(index + 1));
        transition->set_behavior_log_probability(-0.5f - index * 0.1f);
        transition->set_behavior_value(0.2f + index * 0.1f);
        transition->set_advantage(index == 0 ? 0.5f : -0.25f);
        transition->set_value_target(index == 0 ? 0.7f : 0.05f);
        transition->set_behavior_model_step(3);
        transition->set_created_at_unix_ms(1700000000000 + index);
        for (uint32_t action = 0; action < index + 2; ++action) {
            transition->add_action_mask(action != index);
        }
    }
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
                push_response.envelope_id() == envelope.envelope_id(),
            "Push accepts the fixed processed-transition envelope");

    rl::training::v1::PushSamplesRsp duplicate_response;
    pool.Push(push_request, &duplicate_response);
    Require(duplicate_response.result() ==
                rl::training::v1::PUSH_RESULT_DUPLICATE,
            "Push identifies an already accepted envelope by object ID");

    rl::training::v1::GetBatchReq get_request;
    get_request.set_requested_transitions(2);
    get_request.set_timeout_ms(10);
    get_request.set_lease_timeout_ms(1000);
    FillService(get_request.mutable_consumer(), "learner", "learner-fixed");
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

namespace wire = rl::training::v1;

wire::SamplePoolStatusRsp Status(SamplePoolCoordinator& pool) {
    wire::SamplePoolStatusRsp result;
    pool.GetStatus({}, &result);
    return result;
}

wire::GetBatchReq DrawRequest(int count = 2) {
    wire::GetBatchReq request;
    request.set_requested_transitions(count);
    request.set_timeout_ms(1000);
    request.set_lease_timeout_ms(1000);
    FillService(request.mutable_consumer(), "learner", "learner-test");
    return request;
}

void Push(SamplePoolCoordinator& pool, const wire::ProcessedTransitionEnvelope& envelope) {
    wire::PushSamplesReq request;
    *request.mutable_envelope() = envelope;
    wire::PushSamplesRsp response;
    pool.Push(request, &response);
    Require(response.result() == wire::PUSH_RESULT_ACCEPTED,
            "Push accepts test facts: " + response.message());
}

void TestProvenanceSurvivesRedelivery() {
    SamplePoolCoordinator pool(MakeConfig());
    auto first = MakeEnvelope(), second = MakeEnvelope();
    second.set_envelope_id("envelope-other");
    second.mutable_behavior_model()->set_model_lineage_id("other-lineage");
    second.mutable_producer()->set_instance_id("another-producer");
    for (auto& item : *second.mutable_samples()) item.set_item_id("other-" + item.item_id());
    std::map<std::string, wire::SamplePoolItem> expected;
    int64_t expected_bytes = 0;
    for (const auto* envelope : {&first, &second}) {
        Push(pool, *envelope);
        for (const auto& transition : envelope->samples()) {
            auto& item = expected[transition.item_id()];
            *item.mutable_transition() = transition;
            *item.mutable_behavior_model() = envelope->behavior_model();
            *item.mutable_producer() = envelope->producer();
            expected_bytes += transition.ByteSizeLong() + envelope->behavior_model().ByteSizeLong() +
                              envelope->producer().ByteSizeLong();
        }
    }
    auto request = DrawRequest(4);
    request.set_lease_timeout_ms(40);
    for (unsigned draw = 1; draw <= 3; ++draw) {
        wire::GetBatchRsp response;
        pool.GetBatch(request, &response, [] { return false; });
        Require(response.result() == wire::GET_BATCH_RESULT_LEASED && response.items_size() == 4,
                "all source lineages are delivered without Pool admission filtering");
        for (const auto& item : response.items()) {
            const auto& input = expected.at(item.transition().item_id());
            Require(item.transition().SerializeAsString() == input.transition().SerializeAsString() &&
                    item.behavior_model().SerializeAsString() == input.behavior_model().SerializeAsString() &&
                    item.producer().SerializeAsString() == input.producer().SerializeAsString() && item.draw_count() == draw,
                    "item preserves actual model/producer and draw history through NACK and expiry");
        }
        Require(Status(pool).resident_estimated_bytes() == expected_bytes,
                "resident bytes include transition and source fields while leased");
        if (draw == 2) {
            const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (Status(pool).leased_transitions() && std::chrono::steady_clock::now() < limit)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } else {
            wire::NackBatchReq nack;
            *nack.mutable_consumer() = request.consumer();
            nack.set_delivery_id(response.delivery_id());
            wire::DeliveryRsp result;
            pool.Nack(nack, &result);
            Require(result.result() == wire::DELIVERY_RESULT_APPLIED, "NACK restores the lease");
        }
        const auto restored = Status(pool);
        Require(restored.ready_transitions() == 4 && restored.leased_transitions() == 0 &&
                restored.resident_estimated_bytes() == expected_bytes,
                "READY/resident count and bytes survive lease restoration");
    }
}

void TestCancellationWinsReadyAndRestoresDraw() {
    for (bool cancel_after_admission : {false, true}) {
        auto config = MakeConfig();
        config.capacity_transitions = 2;
        SamplePoolCoordinator pool(config);
        Push(pool, MakeEnvelope());
        const auto before = Status(pool);
        bool cancelled = !cancel_after_admission;
        wire::GetBatchRsp response;
        pool.GetBatch(DrawRequest(1), &response, [&] { return std::exchange(cancelled, true); });
        const auto after = Status(pool);
        Require(response.result() == wire::GET_BATCH_RESULT_TIMEOUT && response.delivery_id().empty() &&
                after.ready_transitions() == before.ready_transitions() &&
                after.resident_estimated_bytes() == before.resident_estimated_bytes() &&
                after.leased_transitions() == 0 && after.drawn_transition_slot_count() == 0,
                "cancellation before READY or before lease commit does not consume samples");
        auto newer = MakeEnvelope();
        newer.set_envelope_id("new-envelope");
        newer.mutable_samples()->RemoveLast();
        newer.mutable_samples(0)->set_item_id("new-item");
        Push(pool, newer); // Capacity evicts the original oldest item, even after an aborted draw.
        pool.GetBatch(DrawRequest(), &response, [] { return false; });
        std::set<std::string> ids;
        for (const auto& item : response.items()) {
            ids.insert(item.transition().item_id());
            Require(item.draw_count() == 1, "cancelled draw did not increment item draw count");
        }
        Require(ids == std::set<std::string>({"item-1", "new-item"}),
                "cancelled draw retains original FIFO order for eviction");
    }
}

void TestCancellationWhileWaiting() {
    SamplePoolCoordinator pool(MakeConfig());
    std::promise<void> entered;
    std::atomic<bool> cancelled{false};
    auto result = std::async(std::launch::async, [&] {
        wire::GetBatchRsp response;
        bool notified = false;
        pool.GetBatch(DrawRequest(), &response, [&] {
            if (!notified) { entered.set_value(); notified = true; }
            return cancelled.load();
        });
        return response;
    });
    Require(entered.get_future().wait_for(std::chrono::seconds(2)) == std::future_status::ready,
            "GetBatch reached its waiting boundary");
    cancelled.store(true);
    Push(pool, MakeEnvelope());
    Require(result.get().result() == wire::GET_BATCH_RESULT_TIMEOUT && Status(pool).ready_transitions() == 2,
            "READY wakeup cannot grant the cancelled request a lease");
}

void TestGrpcDeadlineReachesCoordinator() {
    SamplePoolCoordinator pool(MakeConfig());
    SamplePoolConsumerServiceImpl service(pool);
    grpc::ServerBuilder builder;
    int port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&service);
    auto server = builder.BuildAndStart();
    Require(server && port, "start actual SamplePool consumer service");
    auto stub = wire::SamplePoolConsumerService::NewStub(grpc::CreateChannel(
        "127.0.0.1:" + std::to_string(port), grpc::InsecureChannelCredentials()));
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::milliseconds(150));
    wire::GetBatchRsp response;
    const auto result = stub->GetBatch(&context, DrawRequest(), &response);
    Require(result.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED && Status(pool).draw_attempt_count() > 0,
            "actual RPC expires while coordinator waits for samples");
    Push(pool, MakeEnvelope());
    server->Shutdown();
    server->Wait();
    Require(Status(pool).leased_transitions() == 0 && Status(pool).ready_transitions() == 2,
            "expired RPC finishes without creating a later hidden lease");
    pool.GetBatch(DrawRequest(), &response, [] { return false; });
    Require(response.result() == wire::GET_BATCH_RESULT_LEASED,
            "subsequent valid request can use the untouched samples");
}

}  // namespace

int main() {
    TestPushGetAck();
    TestProvenanceSurvivesRedelivery();
    TestCancellationWinsReadyAndRestoresDraw();
    TestCancellationWhileWaiting();
    TestGrpcDeadlineReachesCoordinator();
    std::cout << "sample_pool_data_path: PASS" << std::endl;
    return 0;
}
