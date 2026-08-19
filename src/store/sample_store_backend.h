#pragma once

#include "training.pb.h"

#include <cstdint>
#include <deque>
#include <string>

struct SampleBatchFingerprint {
    std::string payload_sha256;
    uint64_t serialized_size = 0;

    bool operator==(const SampleBatchFingerprint& other) const {
        return payload_sha256 == other.payload_sha256 &&
               serialized_size == other.serialized_size;
    }
};

struct StoredFragment {
    rl::training::v1::SampleBatch batch;
    SampleBatchFingerprint fingerprint;
    std::string policy_key;
    int64_t sample_count = 0;
    int64_t estimated_bytes = 0;
};

// Physical storage is intentionally below ingress validation, delivery leases,
// deduplication and acknowledgement accounting. A future backend must preserve
// this interface's FIFO READY order and exact fragment ownership semantics.
class ISampleStoreBackend {
public:
    virtual ~ISampleStoreBackend() = default;

    virtual const char* name() const = 0;
    virtual const std::deque<StoredFragment>& ready() const = 0;
    virtual void PushBack(StoredFragment fragment) = 0;
    virtual void RestoreFront(std::deque<StoredFragment> fragments) = 0;
    virtual StoredFragment EvictOldestReady() = 0;
    virtual std::deque<StoredFragment> ExtractAllReady() = 0;
    virtual std::deque<StoredFragment> ExtractPolicy(
        const std::string& policy_key,
        int64_t minimum_created_at_unix_ms,
        int64_t target_samples,
        int64_t max_samples,
        bool drain_available) = 0;
};

class LocalFragmentStore final : public ISampleStoreBackend {
public:
    const char* name() const override;
    const std::deque<StoredFragment>& ready() const override;
    void PushBack(StoredFragment fragment) override;
    void RestoreFront(std::deque<StoredFragment> fragments) override;
    StoredFragment EvictOldestReady() override;
    std::deque<StoredFragment> ExtractAllReady() override;
    std::deque<StoredFragment> ExtractPolicy(
        const std::string& policy_key,
        int64_t minimum_created_at_unix_ms,
        int64_t target_samples,
        int64_t max_samples,
        bool drain_available) override;

private:
    std::deque<StoredFragment> ready_;
};
