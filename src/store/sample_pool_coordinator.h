#pragma once

#include "config/config_loader.h"
#include "store/sample_store_backend.h"
#include "training.pb.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

class SamplePoolCoordinator {
public:
    explicit SamplePoolCoordinator(const SamplePoolConfig& config);

    void Push(const rl::training::v1::PushSamplesReq& request,
              rl::training::v1::PushSamplesRsp* response);
    void GetBatch(const rl::training::v1::GetBatchReq& request,
                  rl::training::v1::GetBatchRsp* response,
                  const std::function<bool()>& is_cancelled);
    void Ack(const rl::training::v1::AckBatchReq& request,
             rl::training::v1::DeliveryRsp* response);
    void Nack(const rl::training::v1::NackBatchReq& request,
              rl::training::v1::DeliveryRsp* response);
    void RenewLease(const rl::training::v1::RenewLeaseReq& request,
                    rl::training::v1::DeliveryRsp* response);
    void FinalizeSamplePool(
        const rl::training::v1::FinalizeSamplePoolReq& request,
        rl::training::v1::FinalizeSamplePoolRsp* response);
    void GetStatus(const rl::training::v1::SamplePoolStatusReq& request,
                   rl::training::v1::SamplePoolStatusRsp* response);

    const std::string& instance_id() const;

private:
    struct Lease {
        std::string delivery_id;
        std::string consumer_instance_id;
        std::chrono::steady_clock::time_point deadline;
        int64_t deadline_unix_ms = 0;
        std::deque<StoredFragment> fragments;
        int64_t sample_count = 0;
        int64_t estimated_bytes = 0;
    };

    struct DeliveryRecord {
        rl::training::v1::DeliveryResult result =
            rl::training::v1::DELIVERY_RESULT_UNSPECIFIED;
        rl::training::v1::AckDisposition disposition =
            rl::training::v1::ACK_DISPOSITION_UNSPECIFIED;
        std::string train_update_id;
    };

    struct PolicyCounters {
        rl::training::v1::BehaviorPolicyReference behavior_policy;
        int64_t ready_samples = 0;
        int64_t ready_fragments = 0;
        int64_t leased_samples = 0;
        int64_t leased_fragments = 0;
        int64_t acked_samples = 0;
        int64_t acked_fragments = 0;
        int64_t trained_samples = 0;
        int64_t stale_samples = 0;
        int64_t invalid_samples = 0;
        int64_t shutdown_untrained_samples = 0;
    };

    struct PolicyAvailability {
        std::string policy_key;
        int64_t sample_count = 0;
        size_t first_fifo_index = 0;
    };

    static int64_t NowMs();
    static std::string CreateInstanceId(const std::string& prefix);
    static int64_t CountSamples(const rl::training::v1::SampleBatch& batch);
    static int64_t EstimateBytes(const rl::training::v1::SampleBatch& batch);
    static std::string DeterministicSerialize(
        const rl::training::v1::SampleBatch& batch,
        bool clear_payload_digest);
    static std::string Sha256Hex(const std::string& data);
    static SampleBatchFingerprint FingerprintBatch(
        const rl::training::v1::SampleBatch& batch);
    static bool IsSha256(const rl::common::v1::ContentDigest& digest);
    static bool IsServiceIdentityValid(
        const rl::common::v1::ServiceInstanceIdentity& identity);
    static std::string ServiceKey(
        const rl::common::v1::ServiceInstanceIdentity& identity);
    static bool IsBehaviorPolicyReferenceValid(
        const rl::training::v1::BehaviorPolicyReference& identity);
    static bool IsSchemaIdentityValid(
        const rl::common::v1::SchemaIdentity& identity);
    static std::string PolicyKey(
        const rl::training::v1::BehaviorPolicyReference& policy,
        const rl::training::v1::TrainingSemanticsIdentity& semantics);

    bool ContractMatchesConfig(
        const rl::common::v1::ContractIdentity& contract) const;
    bool ValidateSemantics(
        const rl::training::v1::TrainingSemanticsIdentity& semantics,
        std::string* error) const;
    bool ValidateBatchLocked(const rl::training::v1::SampleBatch& batch,
                             std::string* error,
                             rl::training::v1::PushResult* rejection) const;
    bool DeliveryBelongsToInstanceLocked(const std::string& delivery_id) const;
    bool CapacityAllowsLocked(int64_t samples,
                              int64_t fragments,
                              int64_t estimated_bytes) const;
    bool CanMakeCapacityLocked(int64_t samples,
                               int64_t fragments,
                               int64_t estimated_bytes) const;
    void EvictReadyUntilCapacityLocked(int64_t samples,
                                       int64_t fragments,
                                       int64_t estimated_bytes);
    rl::training::v1::PressureState PressureStateLocked() const;

    void FillServiceIdentity(
        rl::common::v1::ServiceInstanceIdentity* identity) const;
    void FillContractIdentity(
        rl::common::v1::ContractIdentity* identity) const;
    void ReclaimExpiredLeaseLocked();
    void RequeueLeaseLocked(bool expired);
    void RememberDeliveryLocked(const std::string& delivery_id,
                                const DeliveryRecord& record);
    void RememberCompletedBatchLocked(
        const std::string& batch_id,
        const SampleBatchFingerprint& fingerprint);
    const DeliveryRecord* DeliveryHistoryLocked(
        const std::string& delivery_id) const;
    void FillDeliveryResponseLocked(
        rl::training::v1::DeliveryRsp* response) const;
    void FillFinalizeResponseLocked(
        rl::training::v1::FinalizeSamplePoolRsp* response) const;

    bool PolicyMatchesFreshnessLocked(
        const rl::training::v1::BehaviorPolicyReference& policy,
        const rl::training::v1::SampleFreshnessPolicy& freshness) const;
    bool FragmentEligibleLocked(
        const StoredFragment& fragment,
        const rl::training::v1::SampleFreshnessPolicy& freshness,
        const rl::training::v1::TrainingSemanticsIdentity& semantics,
        int64_t minimum_created_at_unix_ms) const;
    std::vector<PolicyAvailability> EligibleAvailabilityLocked(
        const rl::training::v1::SampleFreshnessPolicy& freshness,
        const rl::training::v1::TrainingSemanticsIdentity& semantics,
        int64_t minimum_created_at_unix_ms) const;
    std::string OldestEligiblePolicyLocked(
        const rl::training::v1::SampleFreshnessPolicy& freshness,
        const rl::training::v1::TrainingSemanticsIdentity& semantics,
        int64_t minimum_created_at_unix_ms,
        int64_t minimum_samples) const;

    void FillStatusScalarsLocked(
        rl::training::v1::SamplePoolStatusRsp* response) const;
    static void AppendBehaviorSteps(
        const std::vector<PolicyCounters>& policy_snapshot,
        rl::training::v1::SamplePoolStatusRsp* response);

    SamplePoolConfig config_;
    std::string instance_id_;
    std::unique_ptr<ISampleStoreBackend> backend_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool has_lease_ = false;
    Lease lease_;

    std::unordered_map<std::string, SampleBatchFingerprint>
        active_batch_fingerprints_;
    std::unordered_map<std::string, SampleBatchFingerprint>
        completed_batch_fingerprints_;
    std::deque<std::string> completed_batch_order_;
    std::unordered_map<std::string, DeliveryRecord> delivery_history_;
    std::deque<std::string> delivery_history_order_;
    std::map<std::string, PolicyCounters> policy_counters_;
    std::map<std::string, rl::training::v1::BehaviorPolicyReference>
        behavior_policy_by_key_;
    std::map<std::string, rl::training::v1::TrainingSemanticsIdentity>
        training_semantics_by_key_;
    uint64_t next_delivery_sequence_ = 1;

    int64_t ready_samples_ = 0;
    int64_t ready_fragments_ = 0;
    int64_t ready_estimated_bytes_ = 0;
    int64_t resident_samples_ = 0;
    int64_t resident_fragments_ = 0;
    int64_t resident_estimated_bytes_ = 0;

    int64_t push_attempt_count_ = 0;
    int64_t accepted_unique_samples_ = 0;
    int64_t accepted_unique_batches_ = 0;
    int64_t duplicate_push_attempt_count_ = 0;
    int64_t duplicate_sample_attempts_ = 0;
    int64_t rejected_push_attempt_count_ = 0;
    int64_t rejected_sample_attempts_ = 0;
    int64_t acked_unique_samples_ = 0;
    int64_t acked_unique_batches_ = 0;
    int64_t trained_sample_count_ = 0;
    int64_t stale_sample_count_ = 0;
    int64_t invalid_sample_count_ = 0;
    int64_t shutdown_untrained_sample_count_ = 0;
    int64_t redelivery_count_ = 0;
    int64_t nack_count_ = 0;
    int64_t expired_lease_count_ = 0;
    int64_t lease_renew_count_ = 0;
    int64_t target_hit_count_ = 0;
    int64_t partial_get_count_ = 0;
    int64_t empty_timeout_count_ = 0;
    int64_t consumer_busy_count_ = 0;
    int64_t evicted_sample_count_ = 0;
    int64_t evicted_fragment_count_ = 0;

    bool finalized_ = false;
    std::string finalization_id_;
    std::string finalization_consumer_key_;
    int64_t finalized_at_unix_ms_ = 0;
    int64_t finalized_sample_count_ = 0;
    int64_t finalized_fragment_count_ = 0;

    int64_t latest_ack_unix_ms_ = 0;
    std::string last_error_;
};
