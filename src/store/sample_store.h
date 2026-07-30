#pragma once

#include "config/config_loader.h"
#include "maze.pb.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

class SampleStore {
public:
    explicit SampleStore(const DistributorConfig& config);

    void Push(const maze::SampleBatch& batch, maze::PushSamplesRsp* response);
    void GetBatch(const maze::GetBatchReq& request,
                  maze::GetBatchRsp* response,
                  const std::function<bool()>& is_cancelled);
    void Ack(const maze::AckBatchReq& request, maze::DeliveryRsp* response);
    void Nack(const maze::NackBatchReq& request, maze::DeliveryRsp* response);
    void RenewLease(const maze::RenewLeaseReq& request,
                    maze::DeliveryRsp* response);
    void GetStatus(const maze::DistributorStatusReq& request,
                   maze::DistributorStatusRsp* response);

    const std::string& instance_id() const;

private:
    struct StoredBatch {
        maze::SampleBatch batch;
        int64_t sample_count = 0;
        int64_t estimated_bytes = 0;
    };

    struct Lease {
        std::string delivery_id;
        std::string consumer_instance_id;
        int behavior_model_version = -1;
        std::chrono::steady_clock::time_point deadline;
        int64_t deadline_ts_ms = 0;
        std::deque<StoredBatch> batches;
        int64_t sample_count = 0;
        int64_t estimated_bytes = 0;
    };

    struct DeliveryRecord {
        maze::DeliveryResult result = maze::DELIVERY_RESULT_UNSPECIFIED;
        maze::AckDisposition disposition =
            maze::ACK_DISPOSITION_UNSPECIFIED;
        std::string train_update_id;
    };

    struct VersionCounters {
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

    static constexpr uint32_t kProtocolVersion = 3;

    static int64_t NowMs();
    static std::string CreateInstanceId(const std::string& prefix);
    static int64_t CountSamples(const maze::SampleBatch& batch);
    static int64_t EstimateBytes(const maze::SampleBatch& batch);
    static bool IsSha256(const std::string& value);

    bool ValidateBatchLocked(const maze::SampleBatch& batch,
                             std::string* error) const;
    bool DeliveryBelongsToInstanceLocked(
        const std::string& delivery_id) const;
    bool CapacityAllowsLocked(int64_t samples,
                              int64_t fragments,
                              int64_t estimated_bytes) const;
    maze::PressureState PressureStateLocked() const;

    void ReclaimExpiredLeaseLocked();
    void RequeueLeaseLocked(bool expired);
    void RememberDeliveryLocked(const std::string& delivery_id,
                                const DeliveryRecord& record);
    const DeliveryRecord* DeliveryHistoryLocked(
        const std::string& delivery_id) const;
    void FillStatusLocked(maze::DistributorStatusRsp* response) const;
    void FillDeliveryResponseLocked(maze::DeliveryRsp* response) const;
    int64_t ReadySamplesForVersionLocked(int version) const;

    DistributorConfig config_;
    std::string instance_id_;

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::map<int, std::deque<StoredBatch>> ready_by_version_;
    bool has_lease_ = false;
    Lease lease_;

    std::unordered_set<std::string> accepted_batch_ids_;
    std::unordered_map<std::string, DeliveryRecord> delivery_history_;
    std::deque<std::string> delivery_history_order_;
    std::map<int, VersionCounters> version_counters_;
    uint64_t next_delivery_seq_ = 1;

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

    int64_t latest_push_ts_ms_ = 0;
    int64_t latest_consume_ts_ms_ = 0;
    int64_t latest_ack_ts_ms_ = 0;
    std::string last_error_;
};
