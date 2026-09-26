#pragma once

#include "proto/training/training.pb.h"

#include <cstdint>
#include <deque>
#include <random>
#include <string>
#include <vector>

struct StoredTransition {
    rl::training::v1::ProcessedTransition transition;
    rl::training::v1::ModelIdentity behavior_model;
    rl::common::v1::ServiceInstanceIdentity producer;
    std::string envelope_id;
    uint64_t insert_sequence = 0;
    int64_t inserted_at_unix_ms = 0;
    int64_t estimated_bytes = 0;
    uint32_t draw_count = 0;
};

// Resident replay storage is independent of batch delivery settlement.
// Selection is read-only until commit, so cancellation changes no item facts.
// SnapshotDraw records committed draw slots and returns independent copies.
class ISampleStoreBackend {
public:
    virtual ~ISampleStoreBackend() = default;

    virtual const char* name() const = 0;
    virtual const std::deque<StoredTransition>& resident() const = 0;
    virtual void PushBack(StoredTransition item) = 0;
    virtual StoredTransition EvictOldest() = 0;
    virtual void Clear() = 0;
    virtual std::vector<size_t> SelectUniform(
        size_t count, bool with_replacement,
        std::mt19937_64* generator) const = 0;
    virtual std::vector<StoredTransition> SnapshotDraw(
        const std::vector<size_t>& indices) = 0;
};

class LocalTransitionStore final : public ISampleStoreBackend {
public:
    const char* name() const override;
    const std::deque<StoredTransition>& resident() const override;
    void PushBack(StoredTransition item) override;
    StoredTransition EvictOldest() override;
    void Clear() override;
    std::vector<size_t> SelectUniform(
        size_t count, bool with_replacement,
        std::mt19937_64* generator) const override;
    std::vector<StoredTransition> SnapshotDraw(
        const std::vector<size_t>& indices) override;

private:
    std::deque<StoredTransition> resident_;
};
