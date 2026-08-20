#pragma once

#include "training.pb.h"

#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>

struct EnvelopeFingerprint {
    std::string payload_sha256;
    uint64_t serialized_size = 0;

    bool operator==(const EnvelopeFingerprint& other) const {
        return payload_sha256 == other.payload_sha256 &&
               serialized_size == other.serialized_size;
    }
};

struct StoredTransition {
    rl::training::v1::ProcessedTransition transition;
    rl::training::v1::TrainingSemanticsIdentity training_semantics;
    std::string envelope_id;
    std::string semantics_key;
    std::string profile_digest_hex;
    uint64_t insert_sequence = 0;
    int64_t inserted_at_unix_ms = 0;
    int64_t estimated_bytes = 0;
    uint32_t draw_count = 0;
};

// Storage is below contract validation, envelope deduplication, delivery
// leases and acknowledgement accounting. A future Reverb backend must expose
// these transition-level capabilities without changing transport semantics.
class ISampleStoreBackend {
public:
    virtual ~ISampleStoreBackend() = default;

    virtual const char* name() const = 0;
    virtual const std::deque<StoredTransition>& ready() const = 0;
    virtual void PushBack(StoredTransition item) = 0;
    virtual void RestoreReady(std::vector<StoredTransition> items) = 0;
    virtual StoredTransition EvictOldestReady() = 0;
    virtual std::vector<StoredTransition> ExtractAllReady() = 0;
    virtual std::vector<StoredTransition> DrawUniformWithoutReplacement(
        size_t count,
        const std::string& semantics_key,
        const std::string& profile_digest_hex,
        std::mt19937_64* generator) = 0;
};

class LocalTransitionStore final : public ISampleStoreBackend {
public:
    const char* name() const override;
    const std::deque<StoredTransition>& ready() const override;
    void PushBack(StoredTransition item) override;
    void RestoreReady(std::vector<StoredTransition> items) override;
    StoredTransition EvictOldestReady() override;
    std::vector<StoredTransition> ExtractAllReady() override;
    std::vector<StoredTransition> DrawUniformWithoutReplacement(
        size_t count,
        const std::string& semantics_key,
        const std::string& profile_digest_hex,
        std::mt19937_64* generator) override;

private:
    std::deque<StoredTransition> ready_;
};
