#pragma once

#include "config/config_loader.h"
#include "training.pb.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>

class SampleStore {
public:
    explicit SampleStore(const DistributorConfig& config);

    void UpsertDemand(const rl::training::v1::UpsertSampleDemandReq& request,
                      rl::training::v1::SampleDemandRsp* response);
    void ReleaseDemand(const rl::training::v1::ReleaseSampleDemandReq& request,
                       rl::training::v1::SampleDemandRsp* response);
    void GetDemandStatus(
        const rl::training::v1::GetSampleDemandStatusReq& request,
        rl::training::v1::SampleDemandStatusRsp* response);
    void AcquireCredit(
        const rl::training::v1::AcquireSampleCreditReq& request,
        rl::training::v1::SampleCreditGrant* response);
    void ReleaseCredit(
        const rl::training::v1::ReleaseSampleCreditReq& request,
        rl::training::v1::ReleaseSampleCreditRsp* response);
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
    void GetStatus(const rl::training::v1::DistributorStatusReq& request,
                   rl::training::v1::DistributorStatusRsp* response);

    const std::string& instance_id() const;

private:
    struct BatchFingerprint {
        std::string payload_sha256;
        uint64_t serialized_size = 0;

        bool operator==(const BatchFingerprint& other) const {
            return payload_sha256 == other.payload_sha256 &&
                   serialized_size == other.serialized_size;
        }
    };

    struct StoredBatch {
        rl::training::v1::SampleBatch batch;
        BatchFingerprint fingerprint;
        std::string policy_key;
        int64_t sample_count = 0;
        int64_t estimated_bytes = 0;
    };

    struct Lease {
        std::string delivery_id;
        std::string consumer_instance_id;
        std::chrono::steady_clock::time_point deadline;
        int64_t deadline_unix_ms = 0;
        std::deque<StoredBatch> batches;
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

    struct DemandState {
        rl::training::v1::SampleDemand demand;
        std::string immutable_identity;
        int64_t created_at_unix_ms = 0;
    };

    struct CreditRecord {
        rl::training::v1::AcquireSampleCreditReq request;
        std::string request_identity;
        std::string credit_id;
        std::string demand_id;
        uint64_t demand_epoch = 0;
        int64_t expires_at_unix_ms = 0;
        rl::training::v1::SampleCreditState state =
            rl::training::v1::SAMPLE_CREDIT_STATE_UNSPECIFIED;
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

    static int64_t NowMs();
    static std::string CreateInstanceId(const std::string& prefix);
    static int64_t CountSamples(const rl::training::v1::SampleBatch& batch);
    static int64_t EstimateBytes(const rl::training::v1::SampleBatch& batch);
    static std::string DeterministicSerialize(
        const rl::training::v1::SampleBatch& batch,
        bool clear_payload_digest);
    static std::string Sha256Hex(const std::string& data);
    static BatchFingerprint FingerprintBatch(
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
                             std::string* error) const;
    bool DeliveryBelongsToInstanceLocked(const std::string& delivery_id) const;
    bool CapacityAllowsLocked(int64_t samples,
                              int64_t fragments,
                              int64_t estimated_bytes) const;
    bool DemandWindowAllowsLocked(int64_t samples,
                                  int64_t fragments,
                                  int64_t estimated_bytes) const;
    rl::training::v1::PressureState PressureStateLocked() const;

    bool ValidateDemandLocked(const rl::training::v1::SampleDemand& demand,
                              std::string* error) const;
    bool ValidateCreditRequestLocked(
        const rl::training::v1::AcquireSampleCreditReq& request,
        std::string* error) const;
    bool CreditMatchesBatchLocked(
        const CreditRecord& credit,
        const rl::training::v1::SampleBatch& batch,
        std::string* error) const;
    static std::string DemandImmutableIdentity(
        const rl::training::v1::SampleDemand& demand);
    static std::string CreditRequestIdentity(
        const rl::training::v1::AcquireSampleCreditReq& request);
    void ExpireFlowControlLocked();
    void RevokeCreditsLocked();
    void ReleaseReservationLocked(CreditRecord* credit,
                                  rl::training::v1::SampleCreditState state);
    void TrimTerminalCreditHistoryLocked();
    void FillDemandResponseLocked(
        rl::training::v1::SampleDemandRsp* response) const;
    void FillCreditGrantLocked(
        const CreditRecord& credit,
        rl::training::v1::SampleCreditGrant* response) const;

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
        const BatchFingerprint& fingerprint);
    const DeliveryRecord* DeliveryHistoryLocked(
        const std::string& delivery_id) const;
    void FillStatusLocked(rl::training::v1::DistributorStatusRsp* response) const;
    void FillDeliveryResponseLocked(
        rl::training::v1::DeliveryRsp* response) const;
    bool PolicyMatchesFreshnessLocked(
        const rl::training::v1::BehaviorPolicyReference& policy,
        const rl::training::v1::SampleFreshnessPolicy& freshness) const;
    int64_t ReadySamplesForFreshnessLocked(
        const rl::training::v1::SampleFreshnessPolicy& freshness,
        const rl::training::v1::TrainingSemanticsIdentity& semantics) const;
    std::string OldestCompatiblePolicyKeyLocked(
        const rl::training::v1::SampleFreshnessPolicy& freshness,
        const rl::training::v1::TrainingSemanticsIdentity& semantics) const;
    void ExpireStaleReadyLocked(
        const rl::training::v1::SampleFreshnessPolicy& freshness,
        const rl::training::v1::TrainingSemanticsIdentity& semantics);

    DistributorConfig config_;
    std::string instance_id_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::map<std::string, std::deque<StoredBatch>> ready_by_policy_;
    std::map<std::string, rl::training::v1::BehaviorPolicyReference>
        behavior_policy_by_key_;
    std::map<std::string, rl::training::v1::TrainingSemanticsIdentity>
        training_semantics_by_key_;
    bool has_lease_ = false;
    Lease lease_;

    std::unordered_map<std::string, BatchFingerprint> active_batch_fingerprints_;
    std::unordered_map<std::string, BatchFingerprint> completed_batch_fingerprints_;
    std::deque<std::string> completed_batch_order_;
    std::unordered_map<std::string, DeliveryRecord> delivery_history_;
    std::deque<std::string> delivery_history_order_;
    std::map<std::string, PolicyCounters> policy_counters_;
    uint64_t next_delivery_seq_ = 1;
    uint64_t next_credit_seq_ = 1;

    bool has_active_demand_ = false;
    bool draining_ = false;
    DemandState active_demand_;
    std::unordered_map<std::string, CreditRecord> credits_by_id_;
    std::unordered_map<std::string, std::string> credit_id_by_request_id_;
    std::deque<std::string> terminal_credit_order_;
    int64_t reserved_samples_ = 0;
    int64_t reserved_fragments_ = 0;
    int64_t reserved_estimated_bytes_ = 0;

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
    int64_t credit_request_count_ = 0;
    int64_t credit_grant_count_ = 0;
    int64_t credit_commit_count_ = 0;
    int64_t credit_release_count_ = 0;
    int64_t credit_expire_count_ = 0;
    int64_t credit_revoke_count_ = 0;
    int64_t credit_wait_no_demand_count_ = 0;
    int64_t credit_wait_inflight_limit_count_ = 0;
    int64_t credit_wait_capacity_count_ = 0;
    int64_t credit_wait_draining_count_ = 0;
    int64_t demand_upsert_count_ = 0;
    int64_t demand_release_count_ = 0;

    int64_t latest_push_unix_ms_ = 0;
    int64_t latest_consume_unix_ms_ = 0;
    int64_t latest_ack_unix_ms_ = 0;
    std::string last_error_;
};
