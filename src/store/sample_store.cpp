#include "store/sample_store.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>

SampleStore::SampleStore(const DistributorConfig& config)
    : config_(config),
      instance_id_(CreateInstanceId("distributor")) {}

int64_t SampleStore::NowMs() {
    const auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
        .count();
}

std::string SampleStore::CreateInstanceId(const std::string& prefix) {
    std::random_device rd;
    std::mt19937_64 rng(rd());
    std::ostringstream out;
    out << prefix << "-" << NowMs() << "-"
        << std::hex << std::setw(16) << std::setfill('0') << rng();
    return out.str();
}

int64_t SampleStore::CountSamples(const maze::SampleBatch& batch) {
    return static_cast<int64_t>(batch.samples_size());
}

int64_t SampleStore::EstimateBytes(const maze::SampleBatch& batch) {
    return static_cast<int64_t>(batch.SpaceUsedLong());
}

bool SampleStore::IsSha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (const char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool SampleStore::ValidateBatchLocked(const maze::SampleBatch& batch,
                                      std::string* error) const {
    if (batch.protocol_version() != kProtocolVersion) {
        *error = "unsupported protocol_version";
        return false;
    }
    if (batch.batch_id().empty() || batch.producer_instance_id().empty()) {
        *error = "batch_id and producer_instance_id are required";
        return false;
    }
    if (batch.behavior_model_version() < 0 ||
        !IsSha256(batch.behavior_model_checksum())) {
        *error = "behavior model identity is invalid";
        return false;
    }
    if (batch.samples_size() == 0) {
        *error = "empty fragment is invalid";
        return false;
    }
    if (!batch.bootstrap_valid() || !std::isfinite(batch.bootstrap_value())) {
        *error = "fragment bootstrap is missing or invalid";
        return false;
    }
    if (batch.first_action_frame_id() < 0 ||
        batch.last_action_frame_id() < batch.first_action_frame_id() ||
        batch.last_action_frame_id() - batch.first_action_frame_id() + 1 !=
            batch.samples_size()) {
        *error = "fragment frame range does not match sample count";
        return false;
    }

    for (int index = 0; index < batch.samples_size(); ++index) {
        const auto& sample = batch.samples(index);
        const int64_t expected_frame =
            batch.first_action_frame_id() + index;
        if (sample.action_frame_id() != expected_frame) {
            *error = "sample action_frame_id is not contiguous";
            return false;
        }
        if (sample.obs_size() == 0 || !std::isfinite(sample.reward()) ||
            !std::isfinite(sample.old_log_prob()) ||
            !std::isfinite(sample.old_vpred()) ||
            (sample.terminated() && sample.truncated())) {
            *error = "sample payload is invalid";
            return false;
        }
        if (index + 1 < batch.samples_size() &&
            (sample.terminated() || sample.truncated())) {
            *error = "only the final sample may terminate a fragment";
            return false;
        }
    }

    const auto& final_sample = batch.samples(batch.samples_size() - 1);
    if (batch.is_episode_end() !=
        (final_sample.terminated() || final_sample.truncated())) {
        *error = "episode-end flag does not match final sample";
        return false;
    }
    if (final_sample.termination_reason() ==
            maze::TERMINATION_REASON_GOAL_REACHED &&
        std::fabs(batch.bootstrap_value()) > 1e-6f) {
        *error = "goal-terminated fragment must use zero bootstrap";
        return false;
    }
    if (final_sample.termination_reason() ==
            maze::TERMINATION_REASON_CLIENT_ABORT ||
        final_sample.termination_reason() ==
            maze::TERMINATION_REASON_CHAIN_FAILURE) {
        *error = "aborted fragments must not enter the training pool";
        return false;
    }
    return true;
}

bool SampleStore::DeliveryBelongsToInstanceLocked(
    const std::string& delivery_id) const {
    const std::string prefix = instance_id_ + "-delivery-";
    return delivery_id.rfind(prefix, 0) == 0;
}

bool SampleStore::CapacityAllowsLocked(int64_t samples,
                                       int64_t fragments,
                                       int64_t estimated_bytes) const {
    return resident_samples_ + samples <= config_.max_queue_samples &&
           resident_fragments_ + fragments <= config_.max_queue_fragments &&
           resident_estimated_bytes_ + estimated_bytes <=
               config_.max_queue_estimated_bytes;
}

maze::PressureState SampleStore::PressureStateLocked() const {
    const double sample_ratio =
        static_cast<double>(resident_samples_) / config_.max_queue_samples;
    const double fragment_ratio =
        static_cast<double>(resident_fragments_) /
        config_.max_queue_fragments;
    const double byte_ratio =
        static_cast<double>(resident_estimated_bytes_) /
        config_.max_queue_estimated_bytes;
    const double ratio = std::max({sample_ratio, fragment_ratio, byte_ratio});
    if (ratio >= 1.0) return maze::PRESSURE_STATE_FULL;
    if (ratio >= config_.high_watermark_ratio) {
        return maze::PRESSURE_STATE_HIGH;
    }
    return maze::PRESSURE_STATE_NORMAL;
}

void SampleStore::RememberDeliveryLocked(
    const std::string& delivery_id,
    const DeliveryRecord& record) {
    if (delivery_history_.find(delivery_id) == delivery_history_.end()) {
        delivery_history_order_.push_back(delivery_id);
    }
    delivery_history_[delivery_id] = record;
    while (static_cast<int>(delivery_history_order_.size()) >
           config_.delivery_history_size) {
        const std::string oldest = delivery_history_order_.front();
        delivery_history_order_.pop_front();
        delivery_history_.erase(oldest);
    }
}

const SampleStore::DeliveryRecord* SampleStore::DeliveryHistoryLocked(
    const std::string& delivery_id) const {
    const auto found = delivery_history_.find(delivery_id);
    return found == delivery_history_.end() ? nullptr : &found->second;
}

int64_t SampleStore::ReadySamplesForVersionLocked(int version) const {
    const auto found = version_counters_.find(version);
    return found == version_counters_.end()
               ? 0
               : found->second.ready_samples;
}

void SampleStore::RequeueLeaseLocked(bool expired) {
    if (!has_lease_) return;

    auto& queue = ready_by_version_[lease_.behavior_model_version];
    for (auto it = lease_.batches.rbegin(); it != lease_.batches.rend(); ++it) {
        queue.push_front(std::move(*it));
    }
    ready_samples_ += lease_.sample_count;
    ready_fragments_ += static_cast<int64_t>(lease_.batches.size());
    ready_estimated_bytes_ += lease_.estimated_bytes;
    auto& counters = version_counters_[lease_.behavior_model_version];
    counters.ready_samples += lease_.sample_count;
    counters.ready_fragments += static_cast<int64_t>(lease_.batches.size());
    counters.leased_samples -= lease_.sample_count;
    counters.leased_fragments -=
        static_cast<int64_t>(lease_.batches.size());

    DeliveryRecord record;
    if (expired) {
        ++expired_lease_count_;
        record.result = maze::DELIVERY_RESULT_EXPIRED;
    } else {
        ++nack_count_;
        record.result = maze::DELIVERY_RESULT_APPLIED;
    }
    ++redelivery_count_;
    RememberDeliveryLocked(lease_.delivery_id, record);
    lease_ = Lease{};
    has_lease_ = false;
    cv_.notify_all();
}

void SampleStore::ReclaimExpiredLeaseLocked() {
    if (has_lease_ &&
        std::chrono::steady_clock::now() >= lease_.deadline) {
        RequeueLeaseLocked(true);
    }
}

void SampleStore::Push(const maze::SampleBatch& batch,
                       maze::PushSamplesRsp* response) {
    const int64_t sample_count = CountSamples(batch);
    const int64_t estimated_bytes = EstimateBytes(batch);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ReclaimExpiredLeaseLocked();
        ++push_attempt_count_;
        response->set_batch_id(batch.batch_id());
        response->set_distributor_instance_id(instance_id_);

        std::string error;
        if (!ValidateBatchLocked(batch, &error)) {
            ++rejected_push_attempt_count_;
            rejected_sample_attempts_ += sample_count;
            last_error_ = error;
            response->set_ret_code(-1);
            response->set_message(error);
            response->set_result(maze::PUSH_RESULT_REJECTED_INVALID);
        } else if (accepted_batch_ids_.find(batch.batch_id()) !=
                   accepted_batch_ids_.end()) {
            ++duplicate_push_attempt_count_;
            duplicate_sample_attempts_ += sample_count;
            response->set_ret_code(0);
            response->set_message("already accepted");
            response->set_result(maze::PUSH_RESULT_DUPLICATE);
        } else if (static_cast<int64_t>(accepted_batch_ids_.size()) >=
                       config_.max_dedup_entries ||
                   !CapacityAllowsLocked(
                       sample_count, 1, estimated_bytes)) {
            ++rejected_push_attempt_count_;
            rejected_sample_attempts_ += sample_count;
            last_error_ = "queue or dedup capacity exceeded";
            response->set_ret_code(-1);
            response->set_message(last_error_);
            response->set_result(maze::PUSH_RESULT_REJECTED_CAPACITY);
        } else {
            StoredBatch stored;
            stored.batch = batch;
            stored.sample_count = sample_count;
            stored.estimated_bytes = estimated_bytes;
            ready_by_version_[batch.behavior_model_version()].push_back(
                std::move(stored));
            auto& counters =
                version_counters_[batch.behavior_model_version()];
            counters.ready_samples += sample_count;
            ++counters.ready_fragments;
            ready_samples_ += sample_count;
            ++ready_fragments_;
            ready_estimated_bytes_ += estimated_bytes;
            resident_samples_ += sample_count;
            ++resident_fragments_;
            resident_estimated_bytes_ += estimated_bytes;
            accepted_unique_samples_ += sample_count;
            ++accepted_unique_batches_;
            accepted_batch_ids_.insert(batch.batch_id());
            latest_push_ts_ms_ = NowMs();
            last_error_.clear();

            response->set_ret_code(0);
            response->set_message("accepted");
            response->set_result(maze::PUSH_RESULT_ACCEPTED);
            response->set_accepted_samples(sample_count);
            response->set_accepted_unique_samples(sample_count);
        }

        response->set_queue_size(ready_samples_);
        response->set_resident_samples(resident_samples_);
        response->set_resident_fragments(resident_fragments_);
        response->set_resident_estimated_bytes(resident_estimated_bytes_);
        response->set_pressure_state(PressureStateLocked());
    }
    cv_.notify_all();
}

void SampleStore::GetBatch(
    const maze::GetBatchReq& request,
    maze::GetBatchRsp* response,
    const std::function<bool()>& is_cancelled) {
    const int target_samples = request.batch_size();
    const int timeout_ms = request.timeout_ms() > 0
                               ? request.timeout_ms()
                               : config_.default_get_timeout_ms;
    const int lease_timeout_ms =
        request.lease_timeout_ms() > 0
            ? request.lease_timeout_ms()
            : config_.default_lease_timeout_ms;
    const int version = request.behavior_model_version();
    const auto policy = request.selection_policy();
    const auto start = std::chrono::steady_clock::now();
    const auto wait_deadline =
        start + std::chrono::milliseconds(timeout_ms);

    std::unique_lock<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_distributor_instance_id(instance_id_);
    response->set_behavior_model_version(version);

    if (request.consumer_instance_id().empty() || target_samples <= 0 ||
        timeout_ms <= 0 || lease_timeout_ms <= 0 || version < 0 ||
        (policy != maze::BATCH_SELECTION_POLICY_TARGET_ONLY &&
         policy != maze::BATCH_SELECTION_POLICY_DRAIN_AVAILABLE)) {
        response->set_ret_code(-1);
        response->set_result(maze::GET_BATCH_RESULT_REJECTED);
        response->set_message("invalid GetBatch request");
        response->set_queue_size(ready_samples_);
        return;
    }
    if (has_lease_) {
        ++consumer_busy_count_;
        response->set_ret_code(2);
        response->set_result(maze::GET_BATCH_RESULT_BUSY);
        response->set_message("another delivery is still leased");
        response->set_queue_size(ready_samples_);
        response->set_leased_samples(lease_.sample_count);
        return;
    }

    const auto ready_for_policy = [&]() {
        const int64_t available = ReadySamplesForVersionLocked(version);
        return policy == maze::BATCH_SELECTION_POLICY_TARGET_ONLY
                   ? available >= target_samples
                   : available > 0;
    };
    while (!ready_for_policy() &&
           std::chrono::steady_clock::now() < wait_deadline) {
        if (is_cancelled && is_cancelled()) {
            response->set_ret_code(-1);
            response->set_result(maze::GET_BATCH_RESULT_REJECTED);
            response->set_message("request cancelled");
            response->set_queue_size(ready_samples_);
            return;
        }
        cv_.wait_until(
            lock,
            std::min(
                wait_deadline,
                std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(50)));
        ReclaimExpiredLeaseLocked();
        if (has_lease_) {
            ++consumer_busy_count_;
            response->set_ret_code(2);
            response->set_result(maze::GET_BATCH_RESULT_BUSY);
            response->set_message("another delivery is still leased");
            response->set_queue_size(ready_samples_);
            response->set_leased_samples(lease_.sample_count);
            return;
        }
    }

    if (!ready_for_policy()) {
        ++empty_timeout_count_;
        response->set_ret_code(1);
        response->set_result(maze::GET_BATCH_RESULT_TIMEOUT);
        response->set_message(
            policy == maze::BATCH_SELECTION_POLICY_TARGET_ONLY
                ? "target not available before deadline"
                : "no matching fragment before deadline");
        response->set_queue_size(ready_samples_);
        response->set_wait_ms(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
                .count());
        return;
    }

    Lease new_lease;
    new_lease.consumer_instance_id = request.consumer_instance_id();
    new_lease.behavior_model_version = version;
    new_lease.delivery_id =
        instance_id_ + "-delivery-" + std::to_string(next_delivery_seq_++);
    new_lease.deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(lease_timeout_ms);
    new_lease.deadline_ts_ms = NowMs() + lease_timeout_ms;

    auto& queue = ready_by_version_[version];
    while (!queue.empty() &&
           (new_lease.sample_count < target_samples ||
            new_lease.batches.empty())) {
        StoredBatch stored = std::move(queue.front());
        queue.pop_front();
        ready_samples_ -= stored.sample_count;
        --ready_fragments_;
        ready_estimated_bytes_ -= stored.estimated_bytes;
        auto& counters = version_counters_[version];
        counters.ready_samples -= stored.sample_count;
        --counters.ready_fragments;
        new_lease.sample_count += stored.sample_count;
        new_lease.estimated_bytes += stored.estimated_bytes;
        new_lease.batches.push_back(std::move(stored));
    }

    auto& counters = version_counters_[version];
    counters.leased_samples += new_lease.sample_count;
    counters.leased_fragments +=
        static_cast<int64_t>(new_lease.batches.size());
    if (new_lease.sample_count >= target_samples) {
        ++target_hit_count_;
    } else {
        ++partial_get_count_;
    }
    for (const auto& stored : new_lease.batches) {
        *response->add_batches() = stored.batch;
    }

    response->set_ret_code(0);
    response->set_result(maze::GET_BATCH_RESULT_LEASED);
    response->set_message("leased");
    response->set_delivery_id(new_lease.delivery_id);
    response->set_lease_deadline_ts_ms(new_lease.deadline_ts_ms);
    response->set_returned_samples(new_lease.sample_count);
    response->set_actual_batch_size(new_lease.sample_count);
    response->set_returned_fragments(
        static_cast<int64_t>(new_lease.batches.size()));
    response->set_queue_size(ready_samples_);
    response->set_leased_samples(new_lease.sample_count);
    response->set_wait_ms(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    lease_ = std::move(new_lease);
    has_lease_ = true;
    latest_consume_ts_ms_ = NowMs();
}

void SampleStore::FillDeliveryResponseLocked(
    maze::DeliveryRsp* response) const {
    response->set_queue_size(ready_samples_);
    if (has_lease_) {
        response->set_lease_deadline_ts_ms(lease_.deadline_ts_ms);
    }
}

void SampleStore::Ack(const maze::AckBatchReq& request,
                      maze::DeliveryRsp* response) {
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_delivery_id(request.delivery_id());
    FillDeliveryResponseLocked(response);

    const bool trained =
        request.disposition() == maze::ACK_DISPOSITION_TRAINED;
    if (request.consumer_instance_id().empty() ||
        request.delivery_id().empty() ||
        request.disposition() == maze::ACK_DISPOSITION_UNSPECIFIED ||
        (trained && request.train_update_id().empty())) {
        response->set_ret_code(-1);
        response->set_result(maze::DELIVERY_RESULT_REJECTED);
        response->set_message("invalid AckBatch request");
        return;
    }
    if (!DeliveryBelongsToInstanceLocked(request.delivery_id())) {
        response->set_ret_code(-1);
        response->set_result(maze::DELIVERY_RESULT_REJECTED);
        response->set_message(
            "delivery belongs to another distributor instance");
        return;
    }

    const DeliveryRecord* history =
        DeliveryHistoryLocked(request.delivery_id());
    if (history) {
        if (history->result == maze::DELIVERY_RESULT_EXPIRED) {
            response->set_ret_code(1);
            response->set_result(maze::DELIVERY_RESULT_EXPIRED);
            response->set_message("delivery lease expired");
            return;
        }
        if (history->disposition != request.disposition() ||
            history->train_update_id != request.train_update_id()) {
            response->set_ret_code(-1);
            response->set_result(maze::DELIVERY_RESULT_REJECTED);
            response->set_message("Ack retry conflicts with applied result");
            return;
        }
        response->set_ret_code(0);
        response->set_result(maze::DELIVERY_RESULT_ALREADY_APPLIED);
        response->set_message("delivery already completed");
        response->set_disposition(history->disposition);
        response->set_train_update_id(history->train_update_id);
        return;
    }

    if (!has_lease_ || lease_.delivery_id != request.delivery_id()) {
        response->set_ret_code(1);
        response->set_result(maze::DELIVERY_RESULT_NOT_FOUND);
        response->set_message("delivery not found");
        return;
    }
    if (lease_.consumer_instance_id != request.consumer_instance_id()) {
        response->set_ret_code(-1);
        response->set_result(maze::DELIVERY_RESULT_REJECTED);
        response->set_message("consumer_instance_id does not own delivery");
        return;
    }

    const int64_t affected_samples = lease_.sample_count;
    const int64_t affected_fragments =
        static_cast<int64_t>(lease_.batches.size());
    auto& counters =
        version_counters_[lease_.behavior_model_version];
    counters.leased_samples -= affected_samples;
    counters.leased_fragments -= affected_fragments;
    counters.acked_samples += affected_samples;
    counters.acked_fragments += affected_fragments;
    switch (request.disposition()) {
        case maze::ACK_DISPOSITION_TRAINED:
            counters.trained_samples += affected_samples;
            trained_sample_count_ += affected_samples;
            break;
        case maze::ACK_DISPOSITION_STALE:
            counters.stale_samples += affected_samples;
            stale_sample_count_ += affected_samples;
            break;
        case maze::ACK_DISPOSITION_INVALID:
            counters.invalid_samples += affected_samples;
            invalid_sample_count_ += affected_samples;
            break;
        case maze::ACK_DISPOSITION_SHUTDOWN_UNTRAINED:
            counters.shutdown_untrained_samples += affected_samples;
            shutdown_untrained_sample_count_ += affected_samples;
            break;
        default:
            break;
    }

    resident_samples_ -= affected_samples;
    resident_fragments_ -= affected_fragments;
    resident_estimated_bytes_ -= lease_.estimated_bytes;
    acked_unique_samples_ += affected_samples;
    acked_unique_batches_ += affected_fragments;
    latest_ack_ts_ms_ = NowMs();
    latest_consume_ts_ms_ = latest_ack_ts_ms_;
    DeliveryRecord record;
    record.result = maze::DELIVERY_RESULT_APPLIED;
    record.disposition = request.disposition();
    record.train_update_id = request.train_update_id();
    RememberDeliveryLocked(lease_.delivery_id, record);
    lease_ = Lease{};
    has_lease_ = false;

    response->set_ret_code(0);
    response->set_result(maze::DELIVERY_RESULT_APPLIED);
    response->set_message("acked");
    response->set_affected_samples(affected_samples);
    response->set_disposition(record.disposition);
    response->set_train_update_id(record.train_update_id);
    response->set_queue_size(ready_samples_);
    cv_.notify_all();
}

void SampleStore::Nack(const maze::NackBatchReq& request,
                       maze::DeliveryRsp* response) {
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_delivery_id(request.delivery_id());
    FillDeliveryResponseLocked(response);

    if (request.consumer_instance_id().empty() ||
        request.delivery_id().empty()) {
        response->set_ret_code(-1);
        response->set_result(maze::DELIVERY_RESULT_REJECTED);
        response->set_message("invalid NackBatch request");
        return;
    }
    if (!DeliveryBelongsToInstanceLocked(request.delivery_id())) {
        response->set_ret_code(-1);
        response->set_result(maze::DELIVERY_RESULT_REJECTED);
        response->set_message(
            "delivery belongs to another distributor instance");
        return;
    }
    const DeliveryRecord* history =
        DeliveryHistoryLocked(request.delivery_id());
    if (history) {
        response->set_ret_code(
            history->result == maze::DELIVERY_RESULT_EXPIRED ? 1 : 0);
        response->set_result(
            history->result == maze::DELIVERY_RESULT_EXPIRED
                ? maze::DELIVERY_RESULT_EXPIRED
                : maze::DELIVERY_RESULT_ALREADY_APPLIED);
        response->set_message("delivery already completed");
        return;
    }
    if (!has_lease_ || lease_.delivery_id != request.delivery_id()) {
        response->set_ret_code(1);
        response->set_result(maze::DELIVERY_RESULT_NOT_FOUND);
        response->set_message("delivery not found");
        return;
    }
    if (lease_.consumer_instance_id != request.consumer_instance_id()) {
        response->set_ret_code(-1);
        response->set_result(maze::DELIVERY_RESULT_REJECTED);
        response->set_message("consumer_instance_id does not own delivery");
        return;
    }

    const int64_t affected = lease_.sample_count;
    RequeueLeaseLocked(false);
    response->set_ret_code(0);
    response->set_result(maze::DELIVERY_RESULT_APPLIED);
    response->set_message("nacked and requeued");
    response->set_affected_samples(affected);
    response->set_queue_size(ready_samples_);
}

void SampleStore::RenewLease(const maze::RenewLeaseReq& request,
                             maze::DeliveryRsp* response) {
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_delivery_id(request.delivery_id());
    FillDeliveryResponseLocked(response);

    if (request.consumer_instance_id().empty() ||
        request.delivery_id().empty() || request.lease_timeout_ms() <= 0) {
        response->set_ret_code(-1);
        response->set_result(maze::DELIVERY_RESULT_REJECTED);
        response->set_message("invalid RenewLease request");
        return;
    }
    if (!DeliveryBelongsToInstanceLocked(request.delivery_id())) {
        response->set_ret_code(-1);
        response->set_result(maze::DELIVERY_RESULT_REJECTED);
        response->set_message(
            "delivery belongs to another distributor instance");
        return;
    }
    const DeliveryRecord* history =
        DeliveryHistoryLocked(request.delivery_id());
    if (history) {
        response->set_ret_code(1);
        response->set_result(history->result);
        response->set_message("delivery is already completed");
        return;
    }
    if (!has_lease_ || lease_.delivery_id != request.delivery_id()) {
        response->set_ret_code(1);
        response->set_result(maze::DELIVERY_RESULT_NOT_FOUND);
        response->set_message("delivery not found");
        return;
    }
    if (lease_.consumer_instance_id != request.consumer_instance_id()) {
        response->set_ret_code(-1);
        response->set_result(maze::DELIVERY_RESULT_REJECTED);
        response->set_message("consumer_instance_id does not own delivery");
        return;
    }

    lease_.deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(request.lease_timeout_ms());
    lease_.deadline_ts_ms = NowMs() + request.lease_timeout_ms();
    ++lease_renew_count_;
    response->set_ret_code(0);
    response->set_result(maze::DELIVERY_RESULT_APPLIED);
    response->set_message("lease renewed");
    response->set_affected_samples(lease_.sample_count);
    response->set_lease_deadline_ts_ms(lease_.deadline_ts_ms);
}

void SampleStore::FillStatusLocked(
    maze::DistributorStatusRsp* response) const {
    response->set_push_sample_count(accepted_unique_samples_);
    response->set_consume_sample_count(acked_unique_samples_);
    response->set_queue_size(ready_samples_);
    response->set_push_batch_count(accepted_unique_batches_);
    response->set_consume_batch_count(acked_unique_batches_);
    response->set_drop_count(0);
    response->set_latest_push_ts_ms(latest_push_ts_ms_);
    response->set_latest_consume_ts_ms(latest_consume_ts_ms_);
    response->set_protocol_version(kProtocolVersion);
    response->set_distributor_instance_id(instance_id_);
    response->set_ready(true);
    response->set_push_attempt_count(push_attempt_count_);
    response->set_accepted_unique_samples(accepted_unique_samples_);
    response->set_accepted_unique_batches(accepted_unique_batches_);
    response->set_duplicate_push_attempt_count(
        duplicate_push_attempt_count_);
    response->set_duplicate_sample_attempts(duplicate_sample_attempts_);
    response->set_rejected_push_attempt_count(rejected_push_attempt_count_);
    response->set_rejected_sample_attempts(rejected_sample_attempts_);
    response->set_acked_unique_samples(acked_unique_samples_);
    response->set_acked_unique_batches(acked_unique_batches_);
    response->set_ready_queue_samples(ready_samples_);
    response->set_ready_queue_fragments(ready_fragments_);
    response->set_leased_samples(has_lease_ ? lease_.sample_count : 0);
    response->set_leased_fragments(
        has_lease_ ? static_cast<int64_t>(lease_.batches.size()) : 0);
    response->set_resident_samples(resident_samples_);
    response->set_resident_fragments(resident_fragments_);
    response->set_resident_estimated_bytes(resident_estimated_bytes_);
    response->set_capacity_samples(config_.max_queue_samples);
    response->set_capacity_fragments(config_.max_queue_fragments);
    response->set_capacity_estimated_bytes(
        config_.max_queue_estimated_bytes);
    response->set_pressure_state(PressureStateLocked());
    response->set_redelivery_count(redelivery_count_);
    response->set_nack_count(nack_count_);
    response->set_expired_lease_count(expired_lease_count_);
    response->set_latest_ack_ts_ms(latest_ack_ts_ms_);
    response->set_target_hit_count(target_hit_count_);
    response->set_partial_get_count(partial_get_count_);
    response->set_empty_timeout_count(empty_timeout_count_);
    response->set_last_error(last_error_);
    response->set_trained_sample_count(trained_sample_count_);
    response->set_stale_sample_count(stale_sample_count_);
    response->set_invalid_sample_count(invalid_sample_count_);
    response->set_shutdown_untrained_sample_count(
        shutdown_untrained_sample_count_);
    response->set_lease_renew_count(lease_renew_count_);
    response->set_backend_type(maze::SAMPLE_BACKEND_TYPE_LOCAL_MEMORY);
    response->set_max_concurrent_consumers(1);
    response->set_active_consumer_count(has_lease_ ? 1 : 0);
    response->set_consumer_busy_count(consumer_busy_count_);
    response->set_ingress_ready(true);
    response->set_pool_ready(true);
    for (const auto& item : version_counters_) {
        auto* version = response->add_behavior_versions();
        version->set_behavior_model_version(item.first);
        version->set_ready_samples(item.second.ready_samples);
        version->set_ready_fragments(item.second.ready_fragments);
        version->set_leased_samples(item.second.leased_samples);
        version->set_leased_fragments(item.second.leased_fragments);
        version->set_acked_samples(item.second.acked_samples);
        version->set_acked_fragments(item.second.acked_fragments);
        version->set_trained_samples(item.second.trained_samples);
        version->set_stale_samples(item.second.stale_samples);
        version->set_invalid_samples(item.second.invalid_samples);
        version->set_shutdown_untrained_samples(
            item.second.shutdown_untrained_samples);
    }
}

void SampleStore::GetStatus(const maze::DistributorStatusReq& request,
                            maze::DistributorStatusRsp* response) {
    (void)request;
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    FillStatusLocked(response);
}

const std::string& SampleStore::instance_id() const {
    return instance_id_;
}
