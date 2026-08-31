#include "store/sample_store_backend.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

const char* LocalTransitionStore::name() const {
    return "LocalTransitionStore";
}

const std::deque<StoredTransition>& LocalTransitionStore::ready() const {
    return ready_;
}

void LocalTransitionStore::PushBack(StoredTransition item) {
    ready_.push_back(std::move(item));
}

void LocalTransitionStore::RestoreReady(std::vector<StoredTransition> items) {
    for (auto& item : items) ready_.push_back(std::move(item));
    std::stable_sort(
        ready_.begin(), ready_.end(),
        [](const StoredTransition& left, const StoredTransition& right) {
            return left.insert_sequence < right.insert_sequence;
        });
}

StoredTransition LocalTransitionStore::EvictOldestReady() {
    if (ready_.empty()) {
        throw std::logic_error("cannot evict from an empty transition store");
    }
    StoredTransition item = std::move(ready_.front());
    ready_.pop_front();
    return item;
}

std::vector<StoredTransition> LocalTransitionStore::ExtractAllReady() {
    std::vector<StoredTransition> extracted;
    extracted.reserve(ready_.size());
    while (!ready_.empty()) {
        extracted.push_back(std::move(ready_.front()));
        ready_.pop_front();
    }
    return extracted;
}

std::vector<StoredTransition>
LocalTransitionStore::DrawUniformWithoutReplacement(
    size_t count,
    const std::string& training_contract_digest_hex,
    std::mt19937_64* generator) {
    if (generator == nullptr) {
        throw std::invalid_argument("invalid uniform transition draw");
    }

    std::vector<size_t> indices;
    indices.reserve(ready_.size());
    for (size_t index = 0; index < ready_.size(); ++index) {
        if (ready_[index].training_contract_digest_hex ==
            training_contract_digest_hex) {
            indices.push_back(index);
        }
    }
    if (count > indices.size()) {
        throw std::invalid_argument("insufficient eligible transitions");
    }
    std::shuffle(indices.begin(), indices.end(), *generator);
    indices.resize(count);
    std::sort(indices.begin(), indices.end());

    std::vector<StoredTransition> selected;
    selected.reserve(count);
    std::deque<StoredTransition> remaining;
    size_t selected_cursor = 0;
    for (size_t index = 0; index < ready_.size(); ++index) {
        if (selected_cursor < indices.size() &&
            indices[selected_cursor] == index) {
            selected.push_back(std::move(ready_[index]));
            ++selected_cursor;
        } else {
            remaining.push_back(std::move(ready_[index]));
        }
    }
    ready_ = std::move(remaining);
    return selected;
}
