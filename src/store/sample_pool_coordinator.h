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
#include <random>
#include <set>
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
        std::vector<StoredTransition> items;
        int64_t transition_count = 0;
        int64_t estimated_bytes = 0;
    };

    struct DeliveryRecord {
        rl::training::v1::DeliveryResult result =
            rl::training::v1::DELIVERY_RESULT_UNSPECIFIED;
        rl::training::v1::AckDisposition disposition =
            rl::training::v1::ACK_DISPOSITION_UNSPECIFIED;
        std::string train_update_id;
        int64_t affected_transitions = 0;
    };

    static int64_t NowMs();
    static std::string CreateInstanceId(const std::string& prefix);
    static int64_t EstimateBytes(
        const rl::training::v1::ProcessedTransition& transition);
    static std::string DeterministicSerialize(
        const rl::training::v1::ProcessedTransitionEnvelope& envelope,
        bool clear_payload_digest);
    static std::string Sha256Hex(const std::string& data);
    static EnvelopeFingerprint FingerprintEnvelope(
        const rl::training::v1::ProcessedTransitionEnvelope& envelope);
    static std::string FingerprintTransition(
        const rl::training::v1::ProcessedTransition& transition);
    static bool IsSha256(const rl::common::v1::ContentDigest& digest);
    static bool IsServiceIdentityValid(
        const rl::common::v1::ServiceInstanceIdentity& identity);
    static std::string ServiceKey(
        const rl::common::v1::ServiceInstanceIdentity& identity);
    static bool IsBehaviorPolicyReferenceValid(
        const rl::training::v1::BehaviorPolicyReference& identity);
    static bool IsSchemaIdentityValid(
        const rl::common::v1::SchemaIdentity& identity);
    static std::string SemanticsKey(
        const rl::training::v1::TrainingSemanticsIdentity& semantics);

    bool ContractMatchesConfig(
        const rl::common::v1::ContractIdentity& contract) const;
    bool ValidateSemantics(
        const rl::training::v1::TrainingSemanticsIdentity& semantics,
        std::string* error) const;
    bool ValidateEnvelopeLocked(
        const rl::training::v1::ProcessedTransitionEnvelope& envelope,
        std::string* error,
        rl::training::v1::PushResult* rejection) const;
    bool DeliveryBelongsToInstanceLocked(const std::string& delivery_id) const;
    bool CapacityAllowsLocked(int64_t transitions,
                              int64_t estimated_bytes) const;
    bool CanMakeCapacityLocked(int64_t transitions,
                               int64_t estimated_bytes) const;
    void EvictReadyUntilCapacityLocked(int64_t transitions,
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
    void RememberCompletedEnvelopeLocked(
        const std::string& envelope_id,
        const EnvelopeFingerprint& fingerprint);
    const DeliveryRecord* DeliveryHistoryLocked(
        const std::string& delivery_id) const;
    void FillDeliveryResponseLocked(
        rl::training::v1::DeliveryRsp* response) const;
    void FillFinalizeResponseLocked(
        rl::training::v1::FinalizeSamplePoolRsp* response) const;

    void FillStatusScalarsLocked(
        rl::training::v1::SamplePoolStatusRsp* response) const;
    int64_t EligibleReadyCountLocked(
        const std::string& semantics_key,
        const std::string& profile_digest_hex) const;
    void RemoveResidentItemLocked(const StoredTransition& item);

    SamplePoolConfig config_;
    std::string instance_id_;
    std::unique_ptr<ISampleStoreBackend> backend_;
    std::mt19937_64 random_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool has_lease_ = false;
    Lease lease_;

    std::unordered_map<std::string, EnvelopeFingerprint>
        completed_envelope_fingerprints_;
    std::deque<std::string> completed_envelope_order_;
    std::unordered_map<std::string, std::string> seen_item_fingerprints_;
    std::deque<std::string> seen_item_order_;
    std::set<std::string> resident_item_ids_;
    std::unordered_map<std::string, DeliveryRecord> delivery_history_;
    std::deque<std::string> delivery_history_order_;
    std::unordered_map<std::string, int64_t> resident_by_envelope_;
    uint64_t next_insert_sequence_ = 1;
    uint64_t next_delivery_sequence_ = 1;

    int64_t ready_transitions_ = 0;
    int64_t ready_estimated_bytes_ = 0;
    int64_t resident_transitions_ = 0;
    int64_t resident_estimated_bytes_ = 0;

    int64_t push_attempt_count_ = 0;
    int64_t accepted_unique_transitions_ = 0;
    int64_t accepted_unique_envelopes_ = 0;
    int64_t duplicate_push_attempt_count_ = 0;
    int64_t duplicate_transition_attempts_ = 0;
    int64_t rejected_push_attempt_count_ = 0;
    int64_t rejected_transition_attempts_ = 0;
    int64_t acked_unique_transitions_ = 0;
    int64_t acked_unique_deliveries_ = 0;
    int64_t trained_transition_count_ = 0;
    int64_t invalid_transition_count_ = 0;
    int64_t shutdown_untrained_transition_count_ = 0;
    int64_t redelivery_count_ = 0;
    int64_t nack_count_ = 0;
    int64_t expired_lease_count_ = 0;
    int64_t lease_renew_count_ = 0;
    int64_t target_hit_count_ = 0;
    int64_t draw_attempt_count_ = 0;
    int64_t drawn_transition_slot_count_ = 0;
    int64_t empty_timeout_count_ = 0;
    int64_t consumer_busy_count_ = 0;
    int64_t evicted_transition_count_ = 0;
    int64_t evicted_envelope_count_ = 0;
    int64_t unsampled_evicted_transition_count_ = 0;
    int64_t previously_drawn_evicted_transition_count_ = 0;

    bool finalized_ = false;
    std::string finalization_id_;
    std::string finalization_consumer_key_;
    int64_t finalized_at_unix_ms_ = 0;
    int64_t finalized_transition_count_ = 0;

    int64_t latest_ack_unix_ms_ = 0;
    std::string last_error_;
};
