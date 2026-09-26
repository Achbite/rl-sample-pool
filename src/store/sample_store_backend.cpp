#include "store/sample_store_backend.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>

const char* LocalTransitionStore::name() const {
    return "LocalTransitionStore";
}

const std::deque<StoredTransition>& LocalTransitionStore::resident() const {
    return resident_;
}

void LocalTransitionStore::PushBack(StoredTransition item) {
    resident_.push_back(std::move(item));
}

StoredTransition LocalTransitionStore::EvictOldest() {
    if (resident_.empty()) {
        throw std::logic_error("cannot evict from an empty transition store");
    }
    StoredTransition item = std::move(resident_.front());
    resident_.pop_front();
    return item;
}

void LocalTransitionStore::Clear() {
    resident_.clear();
}

std::vector<size_t> LocalTransitionStore::SelectUniform(
    size_t count, bool with_replacement,
    std::mt19937_64* generator) const {
    if (generator == nullptr || resident_.empty() ||
        (!with_replacement && count > resident_.size())) {
        throw std::invalid_argument("invalid uniform transition draw");
    }
    std::vector<size_t> indices;
    if (with_replacement) {
        indices.reserve(count);
        std::uniform_int_distribution<size_t> select(0, resident_.size() - 1);
        for (size_t slot = 0; slot < count; ++slot) {
            indices.push_back(select(*generator));
        }
    } else {
        indices.resize(resident_.size());
        std::iota(indices.begin(), indices.end(), size_t{0});
        std::shuffle(indices.begin(), indices.end(), *generator);
        indices.resize(count);
        std::sort(indices.begin(), indices.end());
    }
    return indices;
}

std::vector<StoredTransition> LocalTransitionStore::SnapshotDraw(
    const std::vector<size_t>& indices) {
    for (size_t index : indices) ++resident_[index].draw_count;
    std::vector<StoredTransition> selected;
    selected.reserve(indices.size());
    // Repeated slots share the same end-of-draw history and immutable payload.
    for (size_t index : indices) selected.push_back(resident_[index]);
    return selected;
}
