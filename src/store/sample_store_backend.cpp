#include "store/sample_store_backend.h"

#include <stdexcept>
#include <utility>

const char* LocalFragmentStore::name() const {
    return "LocalFragmentStore";
}

const std::deque<StoredFragment>& LocalFragmentStore::ready() const {
    return ready_;
}

void LocalFragmentStore::PushBack(StoredFragment fragment) {
    ready_.push_back(std::move(fragment));
}

void LocalFragmentStore::RestoreFront(std::deque<StoredFragment> fragments) {
    while (!fragments.empty()) {
        ready_.push_front(std::move(fragments.back()));
        fragments.pop_back();
    }
}

StoredFragment LocalFragmentStore::EvictOldestReady() {
    if (ready_.empty()) {
        throw std::logic_error("cannot evict from an empty fragment store");
    }
    StoredFragment fragment = std::move(ready_.front());
    ready_.pop_front();
    return fragment;
}

std::deque<StoredFragment> LocalFragmentStore::ExtractAllReady() {
    std::deque<StoredFragment> extracted;
    extracted.swap(ready_);
    return extracted;
}

std::deque<StoredFragment> LocalFragmentStore::ExtractPolicy(
    const std::string& policy_key,
    int64_t minimum_created_at_unix_ms,
    int64_t target_samples,
    int64_t max_samples,
    bool drain_available) {
    std::deque<StoredFragment> selected;
    std::deque<StoredFragment> remaining;
    int64_t selected_samples = 0;

    while (!ready_.empty()) {
        StoredFragment fragment = std::move(ready_.front());
        ready_.pop_front();
        const bool still_collecting =
            drain_available || selected_samples < target_samples;
        if (still_collecting && fragment.policy_key == policy_key &&
            fragment.batch.created_at_unix_ms() >=
                minimum_created_at_unix_ms &&
            selected_samples + fragment.sample_count <= max_samples) {
            selected_samples += fragment.sample_count;
            selected.push_back(std::move(fragment));
        } else {
            remaining.push_back(std::move(fragment));
        }
    }
    ready_ = std::move(remaining);
    return selected;
}
