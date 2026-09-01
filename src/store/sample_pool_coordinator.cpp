#include "store/sample_pool_coordinator.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <openssl/sha.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

template <typename Message>
std::string DeterministicBytes(const Message& message) {
    std::string output;
    output.reserve(static_cast<size_t>(message.ByteSizeLong()));
    google::protobuf::io::StringOutputStream stream(&output);
    google::protobuf::io::CodedOutputStream coded(&stream);
    coded.SetSerializationDeterministic(true);
    if (!message.SerializeToCodedStream(&coded) || coded.HadError()) {
        throw std::runtime_error("deterministic protobuf serialization failed");
    }
    return output;
}

bool Finite(float value) {
    return std::isfinite(static_cast<double>(value));
}

bool Finite(double value) {
    return std::isfinite(value);
}

}  // namespace

SamplePoolCoordinator::SamplePoolCoordinator(const SamplePoolConfig& config)
    : config_(config),
      instance_id_(CreateInstanceId("sample-pool")),
      backend_(std::make_unique<LocalTransitionStore>()),
      random_(config.sampling_seed) {}

int64_t SamplePoolCoordinator::NowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string SamplePoolCoordinator::CreateInstanceId(
    const std::string& prefix) {
    const auto now = std::chrono::high_resolution_clock::now()
                         .time_since_epoch()
                         .count();
    std::ostringstream output;
    output << prefix << "-" << now << "-" << std::hex
           << std::hash<std::string>{}(prefix + std::to_string(now));
    return output.str();
}

int64_t SamplePoolCoordinator::EstimateBytes(
    const rl::training::v1::ProcessedTransition& transition) {
    const auto bytes = transition.ByteSizeLong();
    if (bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw std::overflow_error("processed transition size exceeds int64");
    }
    return static_cast<int64_t>(bytes);
}

std::string SamplePoolCoordinator::DeterministicSerialize(
    const rl::training::v1::ProcessedTransitionEnvelope& envelope,
    bool clear_payload_digest) {
    rl::training::v1::ProcessedTransitionEnvelope copy(envelope);
    if (clear_payload_digest) copy.clear_payload_digest();
    return DeterministicBytes(copy);
}

std::string SamplePoolCoordinator::Sha256Hex(const std::string& data) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(data.data()),
           data.size(), digest);
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned char byte : digest) {
        output << std::setw(2) << static_cast<int>(byte);
    }
    return output.str();
}

EnvelopeFingerprint SamplePoolCoordinator::FingerprintEnvelope(
    const rl::training::v1::ProcessedTransitionEnvelope& envelope) {
    const std::string bytes = DeterministicSerialize(envelope, true);
    return {Sha256Hex(bytes), static_cast<uint64_t>(bytes.size())};
}

std::string SamplePoolCoordinator::FingerprintTransition(
    const rl::training::v1::ProcessedTransition& transition) {
    return Sha256Hex(DeterministicBytes(transition));
}

bool SamplePoolCoordinator::IsSha256(
    const rl::common::v1::ContentDigest& digest) {
    if (digest.algorithm() != rl::common::v1::DIGEST_ALGORITHM_SHA256 ||
        digest.hex().size() != 64) {
        return false;
    }
    return std::all_of(digest.hex().begin(), digest.hex().end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

bool SamplePoolCoordinator::IsServiceIdentityValid(
    const rl::common::v1::ServiceInstanceIdentity& identity) {
    return !identity.component().empty() && !identity.instance_id().empty() &&
           identity.lifecycle_epoch() > 0;
}

std::string SamplePoolCoordinator::ServiceKey(
    const rl::common::v1::ServiceInstanceIdentity& identity) {
    return identity.component() + "|" + identity.instance_id() + "|" +
           std::to_string(identity.lifecycle_epoch());
}

bool SamplePoolCoordinator::IsModelIdentityValid(
    const rl::training::v1::ModelIdentity& identity) {
    return !identity.model_lineage_id().empty() && identity.has_model_step() &&
           IsSha256(identity.artifact_digest()) &&
           IsSha256(identity.manifest_digest());
}

bool SamplePoolCoordinator::ValidateEnvelopeLocked(
    const rl::training::v1::ProcessedTransitionEnvelope& envelope,
    std::string* error,
    rl::training::v1::PushResult* rejection) const {
    *rejection = rl::training::v1::PUSH_RESULT_REJECTED_INVALID;
    if (envelope.envelope_id().empty() || envelope.samples_size() <= 0) {
        *error = "envelope identity or samples are missing";
        return false;
    }
    if (!IsServiceIdentityValid(envelope.producer()) ||
        envelope.producer().component() != "sample-distributor") {
        *rejection = rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY;
        *error = "producer must be a concrete sample-distributor instance";
        return false;
    }
    if (!IsSha256(envelope.training_contract_digest()) ||
        !IsModelIdentityValid(envelope.behavior_model())) {
        *rejection = rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY;
        *error = "training contract digest or behavior model is invalid";
        return false;
    }
    if (!IsSha256(envelope.payload_digest())) {
        *error = "envelope payload digest is invalid";
        return false;
    }

    EnvelopeFingerprint fingerprint;
    try {
        fingerprint = FingerprintEnvelope(envelope);
    } catch (const std::exception& exception) {
        *error = exception.what();
        return false;
    }
    if (fingerprint.payload_sha256 != envelope.payload_digest().hex()) {
        *rejection = rl::training::v1::PUSH_RESULT_REJECTED_CONFLICT;
        *error = "envelope payload digest mismatch";
        return false;
    }

    std::set<std::string> item_ids;
    // The AIServer owns segment, termination, bootstrap and estimator-profile
    // integrity before it projects these final PPO samples. The pool owns the
    // immutable wire and delivery boundary only.
    for (const auto& item : envelope.samples()) {
        if (item.item_id().empty() || item.observation_size() <= 0 ||
            !item.has_behavior_model_step() ||
            item.behavior_model_step() !=
                envelope.behavior_model().model_step() ||
            item.action() < 0 || item.created_at_unix_ms() <= 0) {
            *error = "processed sample identity or provenance is incomplete";
            return false;
        }
        if (!item_ids.insert(item.item_id()).second) {
            *error = "envelope contains duplicate item identity";
            return false;
        }
        for (float value : item.observation()) {
            if (!Finite(value)) {
                *error = "processed sample observation is non-finite";
                return false;
            }
        }
        if (!Finite(item.behavior_log_probability()) ||
            !Finite(item.behavior_value()) || !Finite(item.advantage()) ||
            !Finite(item.value_target())) {
            *error = "processed sample scalar is non-finite";
            return false;
        }
    }
    return true;
}

bool SamplePoolCoordinator::DeliveryBelongsToInstanceLocked(
    const std::string& delivery_id) const {
    const std::string prefix = instance_id_ + "/delivery-";
    return delivery_id.rfind(prefix, 0) == 0;
}

bool SamplePoolCoordinator::CapacityAllowsLocked(
    int64_t transitions,
    int64_t estimated_bytes) const {
    return transitions >= 0 && estimated_bytes >= 0 &&
           resident_transitions_ <=
               config_.capacity_transitions - transitions &&
           resident_estimated_bytes_ <=
               config_.capacity_bytes - estimated_bytes;
}

bool SamplePoolCoordinator::CanMakeCapacityLocked(
    int64_t transitions,
    int64_t estimated_bytes) const {
    if (transitions > config_.capacity_transitions ||
        estimated_bytes > config_.capacity_bytes) {
        return false;
    }
    return resident_transitions_ - ready_transitions_ <=
               config_.capacity_transitions - transitions &&
           resident_estimated_bytes_ - ready_estimated_bytes_ <=
               config_.capacity_bytes - estimated_bytes;
}

void SamplePoolCoordinator::RemoveResidentItemLocked(
    const StoredTransition& item) {
    --resident_transitions_;
    resident_estimated_bytes_ -= item.estimated_bytes;
    resident_item_ids_.erase(item.transition.item_id());
    auto envelope = resident_by_envelope_.find(item.envelope_id);
    if (envelope != resident_by_envelope_.end()) {
        --envelope->second;
        if (envelope->second == 0) resident_by_envelope_.erase(envelope);
    }
}

void SamplePoolCoordinator::EvictReadyUntilCapacityLocked(
    int64_t transitions,
    int64_t estimated_bytes) {
    while (!CapacityAllowsLocked(transitions, estimated_bytes)) {
        StoredTransition item = backend_->EvictOldestReady();
        --ready_transitions_;
        ready_estimated_bytes_ -= item.estimated_bytes;
        const bool envelope_will_be_empty =
            resident_by_envelope_.count(item.envelope_id) == 1 &&
            resident_by_envelope_.at(item.envelope_id) == 1;
        RemoveResidentItemLocked(item);
        ++evicted_transition_count_;
        if (item.draw_count == 0) {
            ++unsampled_evicted_transition_count_;
        } else {
            ++previously_drawn_evicted_transition_count_;
        }
        if (envelope_will_be_empty) ++evicted_envelope_count_;
        std::cout << "[SamplePool][Eviction] item_id="
                  << item.transition.item_id()
                  << " insert_sequence=" << item.insert_sequence
                  << " inserted_at_unix_ms=" << item.inserted_at_unix_ms
                  << " resident_age_ms=" << (NowMs() - item.inserted_at_unix_ms)
                  << " draw_count=" << item.draw_count
                  << " ever_sampled=" << (item.draw_count > 0 ? "true" : "false")
                  << " reason=FIFO_READY_CAPACITY" << std::endl;
    }
}

rl::training::v1::PressureState
SamplePoolCoordinator::PressureStateLocked() const {
    if (resident_transitions_ >= config_.capacity_transitions ||
        resident_estimated_bytes_ >= config_.capacity_bytes) {
        return rl::training::v1::PRESSURE_STATE_FULL;
    }
    const double transition_ratio =
        static_cast<double>(resident_transitions_) /
        static_cast<double>(config_.capacity_transitions);
    const double byte_ratio =
        static_cast<double>(resident_estimated_bytes_) /
        static_cast<double>(config_.capacity_bytes);
    return std::max(transition_ratio, byte_ratio) >=
                   config_.high_watermark_ratio
               ? rl::training::v1::PRESSURE_STATE_HIGH
               : rl::training::v1::PRESSURE_STATE_NORMAL;
}

void SamplePoolCoordinator::FillServiceIdentity(
    rl::common::v1::ServiceInstanceIdentity* identity) const {
    identity->set_component("sample-pool");
    identity->set_instance_id(instance_id_);
    identity->set_lifecycle_epoch(1);
}

void SamplePoolCoordinator::FillContractIdentity(
    rl::common::v1::ContractIdentity* identity) const {
    identity->set_package_name(config_.contract.package_name);
    identity->set_package_version(config_.contract.package_version);
    identity->set_platform(config_.contract.platform);
}

void SamplePoolCoordinator::RequeueLeaseLocked(bool expired) {
    if (!has_lease_) return;
    const int64_t count = lease_.transition_count;
    backend_->RestoreReady(std::move(lease_.items));
    ready_transitions_ += count;
    ready_estimated_bytes_ += lease_.estimated_bytes;
    if (expired) {
        ++expired_lease_count_;
        redelivery_count_ += count;
    }
    lease_ = Lease{};
    has_lease_ = false;
    cv_.notify_all();
}

void SamplePoolCoordinator::ReclaimExpiredLeaseLocked() {
    if (has_lease_ &&
        std::chrono::steady_clock::now() >= lease_.deadline) {
        RequeueLeaseLocked(true);
    }
}

void SamplePoolCoordinator::RememberDeliveryLocked(
    const std::string& delivery_id,
    const DeliveryRecord& record) {
    delivery_history_[delivery_id] = record;
    delivery_history_order_.push_back(delivery_id);
    while (delivery_history_order_.size() >
           static_cast<size_t>(config_.delivery_history_size)) {
        delivery_history_.erase(delivery_history_order_.front());
        delivery_history_order_.pop_front();
    }
}

void SamplePoolCoordinator::RememberCompletedEnvelopeLocked(
    const std::string& envelope_id,
    const EnvelopeFingerprint& fingerprint) {
    completed_envelope_fingerprints_[envelope_id] = fingerprint;
    completed_envelope_order_.push_back(envelope_id);
    while (completed_envelope_order_.size() >
           static_cast<size_t>(config_.max_dedup_entries)) {
        completed_envelope_fingerprints_.erase(
            completed_envelope_order_.front());
        completed_envelope_order_.pop_front();
    }
}

const SamplePoolCoordinator::DeliveryRecord*
SamplePoolCoordinator::DeliveryHistoryLocked(
    const std::string& delivery_id) const {
    const auto found = delivery_history_.find(delivery_id);
    return found == delivery_history_.end() ? nullptr : &found->second;
}

void SamplePoolCoordinator::FillFinalizeResponseLocked(
    rl::training::v1::FinalizeSamplePoolRsp* response) const {
    response->set_finalization_id(finalization_id_);
    FillServiceIdentity(response->mutable_sample_pool());
    if (finalized_at_unix_ms_ > 0) {
        response->set_finalized_at_unix_ms(finalized_at_unix_ms_);
    }
}

int64_t SamplePoolCoordinator::EligibleReadyCountLocked(
    const std::string& training_contract_digest_hex) const {
    return static_cast<int64_t>(std::count_if(
        backend_->ready().begin(), backend_->ready().end(),
        [&](const StoredTransition& item) {
            return item.training_contract_digest_hex ==
                   training_contract_digest_hex;
        }));
}

void SamplePoolCoordinator::Push(
    const rl::training::v1::PushSamplesReq& request,
    rl::training::v1::PushSamplesRsp* response) {
    std::unique_lock<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    ++push_attempt_count_;

    const auto& envelope = request.envelope();
    const int64_t attempted = envelope.samples_size();
    response->set_envelope_id(envelope.envelope_id());
    FillServiceIdentity(response->mutable_sample_pool());

    if (finalized_) {
        ++rejected_push_attempt_count_;
        rejected_transition_attempts_ += attempted;
        response->set_result(
            rl::training::v1::PUSH_RESULT_REJECTED_FINALIZED);
        response->set_message("SamplePool ingress is finalized");
        response->set_pressure_state(PressureStateLocked());
        return;
    }

    std::string error;
    auto rejection = rl::training::v1::PUSH_RESULT_REJECTED_INVALID;
    if (!ValidateEnvelopeLocked(envelope, &error, &rejection)) {
        ++rejected_push_attempt_count_;
        rejected_transition_attempts_ += attempted;
        last_error_ = error;
        response->set_result(rejection);
        response->set_message(error);
        response->set_pressure_state(PressureStateLocked());
        return;
    }

    const EnvelopeFingerprint envelope_fingerprint =
        FingerprintEnvelope(envelope);
    const auto existing_envelope =
        completed_envelope_fingerprints_.find(envelope.envelope_id());
    if (existing_envelope != completed_envelope_fingerprints_.end()) {
        if (existing_envelope->second == envelope_fingerprint) {
            ++duplicate_push_attempt_count_;
            duplicate_transition_attempts_ += attempted;
            response->set_result(rl::training::v1::PUSH_RESULT_DUPLICATE);
            response->set_message("envelope already accepted");
            response->mutable_payload_digest()->CopyFrom(
                envelope.payload_digest());
        } else {
            ++rejected_push_attempt_count_;
            rejected_transition_attempts_ += attempted;
            last_error_ = "envelope identity reused with different bytes";
            response->set_result(
                rl::training::v1::PUSH_RESULT_REJECTED_CONFLICT);
            response->set_message(last_error_);
        }
        response->set_pressure_state(PressureStateLocked());
        return;
    }

    int existing_item_count = 0;
    for (const auto& transition : envelope.samples()) {
        const auto existing =
            seen_item_fingerprints_.find(transition.item_id());
        if (existing == seen_item_fingerprints_.end()) continue;
        ++existing_item_count;
        if (existing->second != FingerprintTransition(transition)) {
            ++rejected_push_attempt_count_;
            rejected_transition_attempts_ += attempted;
            last_error_ =
                "item identity reused with different processed transition";
            response->set_result(
                rl::training::v1::PUSH_RESULT_REJECTED_CONFLICT);
            response->set_message(last_error_);
            response->set_pressure_state(PressureStateLocked());
            return;
        }
    }
    if (existing_item_count > 0) {
        if (existing_item_count == attempted) {
            ++duplicate_push_attempt_count_;
            duplicate_transition_attempts_ += attempted;
            RememberCompletedEnvelopeLocked(
                envelope.envelope_id(), envelope_fingerprint);
            response->set_result(rl::training::v1::PUSH_RESULT_DUPLICATE);
            response->set_message("all envelope items were already accepted");
            response->mutable_payload_digest()->CopyFrom(
                envelope.payload_digest());
        } else {
            ++rejected_push_attempt_count_;
            rejected_transition_attempts_ += attempted;
            last_error_ =
                "envelope partially overlaps previously accepted item identities";
            response->set_result(
                rl::training::v1::PUSH_RESULT_REJECTED_CONFLICT);
            response->set_message(last_error_);
        }
        response->set_pressure_state(PressureStateLocked());
        return;
    }

    int64_t incoming_bytes = 0;
    try {
        for (const auto& transition : envelope.samples()) {
            const int64_t bytes = EstimateBytes(transition);
            if (incoming_bytes >
                std::numeric_limits<int64_t>::max() - bytes) {
                throw std::overflow_error("envelope byte total overflow");
            }
            incoming_bytes += bytes;
        }
    } catch (const std::exception& exception) {
        ++rejected_push_attempt_count_;
        rejected_transition_attempts_ += attempted;
        last_error_ = exception.what();
        response->set_result(
            rl::training::v1::PUSH_RESULT_REJECTED_INVALID);
        response->set_message(last_error_);
        return;
    }

    if (!CanMakeCapacityLocked(attempted, incoming_bytes)) {
        ++rejected_push_attempt_count_;
        rejected_transition_attempts_ += attempted;
        last_error_ = "insufficient evictable READY capacity";
        response->set_result(
            rl::training::v1::PUSH_RESULT_REJECTED_CAPACITY);
        response->set_message(last_error_);
        response->set_pressure_state(PressureStateLocked());
        return;
    }
    EvictReadyUntilCapacityLocked(attempted, incoming_bytes);

    const int64_t inserted_at = NowMs();
    for (const auto& transition : envelope.samples()) {
        StoredTransition stored;
        stored.transition = transition;
        stored.envelope_id = envelope.envelope_id();
        stored.training_contract_digest_hex =
            envelope.training_contract_digest().hex();
        stored.insert_sequence = next_insert_sequence_++;
        stored.inserted_at_unix_ms = inserted_at;
        stored.estimated_bytes = EstimateBytes(transition);
        backend_->PushBack(std::move(stored));
        ++ready_transitions_;
        ++resident_transitions_;
        ready_estimated_bytes_ += EstimateBytes(transition);
        resident_estimated_bytes_ += EstimateBytes(transition);
        resident_item_ids_.insert(transition.item_id());
        seen_item_fingerprints_[transition.item_id()] =
            FingerprintTransition(transition);
        seen_item_order_.push_back(transition.item_id());
    }
    resident_by_envelope_[envelope.envelope_id()] += attempted;
    RememberCompletedEnvelopeLocked(
        envelope.envelope_id(), envelope_fingerprint);

    size_t examined = seen_item_order_.size();
    while (seen_item_fingerprints_.size() >
               static_cast<size_t>(config_.max_dedup_entries) &&
           examined-- > 0) {
        const std::string item_id = seen_item_order_.front();
        seen_item_order_.pop_front();
        if (resident_item_ids_.count(item_id) > 0) {
            seen_item_order_.push_back(item_id);
        } else {
            seen_item_fingerprints_.erase(item_id);
        }
    }

    accepted_unique_transitions_ += attempted;
    ++accepted_unique_envelopes_;
    response->set_result(rl::training::v1::PUSH_RESULT_ACCEPTED);
    response->set_message("accepted");
    response->set_pressure_state(PressureStateLocked());
    response->mutable_payload_digest()->CopyFrom(envelope.payload_digest());
    cv_.notify_all();
}

void SamplePoolCoordinator::GetBatch(
    const rl::training::v1::GetBatchReq& request,
    rl::training::v1::GetBatchRsp* response,
    const std::function<bool()>& is_cancelled) {
    const auto started = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    FillServiceIdentity(response->mutable_sample_pool());

    if (request.requested_transitions() <= 0 ||
        request.timeout_ms() <= 0 || request.lease_timeout_ms() <= 0 ||
        !IsServiceIdentityValid(request.consumer()) ||
        !IsSha256(request.required_training_contract_digest())) {
        response->set_result(rl::training::v1::GET_BATCH_RESULT_REJECTED);
        response->set_message("draw request is invalid");
        return;
    }

    const std::string consumer_key = ServiceKey(request.consumer());
    const std::string training_contract_digest =
        request.required_training_contract_digest().hex();
    ++draw_attempt_count_;
    const auto timeout =
        std::chrono::milliseconds(request.timeout_ms());
    const auto deadline = started + timeout;

    while (true) {
        ReclaimExpiredLeaseLocked();
        if (has_lease_) {
            ++consumer_busy_count_;
            response->set_result(rl::training::v1::GET_BATCH_RESULT_BUSY);
            response->set_message("single consumer lease is active");
            return;
        }
        if (EligibleReadyCountLocked(training_contract_digest) >=
            request.requested_transitions()) {
            break;
        }
        if (is_cancelled() || std::chrono::steady_clock::now() >= deadline) {
            ++empty_timeout_count_;
            response->set_result(rl::training::v1::GET_BATCH_RESULT_TIMEOUT);
            response->set_message("requested transition count is not ready");
            return;
        }
        cv_.wait_until(lock, deadline);
    }

    std::vector<StoredTransition> selected =
        backend_->DrawUniformWithoutReplacement(
            static_cast<size_t>(request.requested_transitions()),
            training_contract_digest, &random_);
    const int64_t leased_at = NowMs();
    const int64_t lease_timeout_ms = request.lease_timeout_ms();
    lease_.delivery_id = instance_id_ + "/delivery-" +
                         std::to_string(next_delivery_sequence_++);
    lease_.consumer_instance_id = consumer_key;
    lease_.deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(lease_timeout_ms);
    lease_.deadline_unix_ms = leased_at + lease_timeout_ms;
    lease_.transition_count = selected.size();
    lease_.estimated_bytes = 0;

    for (auto& item : selected) {
        ++item.draw_count;
        lease_.estimated_bytes += item.estimated_bytes;
        auto* output = response->add_items();
        *output->mutable_transition() = item.transition;
        output->set_insert_sequence(item.insert_sequence);
        output->set_inserted_at_unix_ms(item.inserted_at_unix_ms);
        output->set_draw_count(item.draw_count);
    }
    lease_.items = std::move(selected);
    drawn_transition_slot_count_ += lease_.transition_count;
    ready_transitions_ -= lease_.transition_count;
    ready_estimated_bytes_ -= lease_.estimated_bytes;
    has_lease_ = true;
    ++target_hit_count_;

    response->set_result(rl::training::v1::GET_BATCH_RESULT_LEASED);
    response->set_message("leased");
    response->set_delivery_id(lease_.delivery_id);
    response->set_lease_deadline_unix_ms(lease_.deadline_unix_ms);
}

void SamplePoolCoordinator::Ack(
    const rl::training::v1::AckBatchReq& request,
    rl::training::v1::DeliveryRsp* response) {
    std::unique_lock<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_delivery_id(request.delivery_id());

    if (!IsServiceIdentityValid(request.consumer()) ||
        request.delivery_id().empty() ||
        !DeliveryBelongsToInstanceLocked(request.delivery_id())) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("delivery or consumer identity is invalid");
        return;
    }
    if (const DeliveryRecord* history =
            DeliveryHistoryLocked(request.delivery_id())) {
        response->set_result(
            rl::training::v1::DELIVERY_RESULT_ALREADY_APPLIED);
        response->set_message("delivery settlement already applied");
        response->set_disposition(history->disposition);
        response->set_train_update_id(history->train_update_id);
        return;
    }
    if (!has_lease_ || lease_.delivery_id != request.delivery_id()) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_NOT_FOUND);
        response->set_message("active delivery not found");
        return;
    }
    if (lease_.consumer_instance_id != ServiceKey(request.consumer())) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("delivery belongs to another consumer");
        return;
    }
    const auto disposition = request.disposition();
    if (disposition != rl::training::v1::ACK_DISPOSITION_TRAINED &&
        disposition != rl::training::v1::ACK_DISPOSITION_INVALID &&
        disposition !=
            rl::training::v1::ACK_DISPOSITION_SHUTDOWN_UNTRAINED) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("ack disposition is invalid");
        return;
    }
    if (disposition == rl::training::v1::ACK_DISPOSITION_TRAINED &&
        request.train_update_id().empty()) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("trained acknowledgement requires update id");
        return;
    }

    const int64_t affected = lease_.transition_count;
    for (const auto& item : lease_.items) RemoveResidentItemLocked(item);
    if (disposition == rl::training::v1::ACK_DISPOSITION_TRAINED) {
        trained_transition_count_ += affected;
    } else if (disposition == rl::training::v1::ACK_DISPOSITION_INVALID) {
        invalid_transition_count_ += affected;
    } else {
        shutdown_untrained_transition_count_ += affected;
    }
    acked_unique_transitions_ += affected;
    ++acked_unique_deliveries_;
    latest_ack_unix_ms_ = NowMs();

    DeliveryRecord record;
    record.result = rl::training::v1::DELIVERY_RESULT_APPLIED;
    record.disposition = disposition;
    record.train_update_id = request.train_update_id();
    RememberDeliveryLocked(request.delivery_id(), record);
    lease_ = Lease{};
    has_lease_ = false;

    response->set_result(rl::training::v1::DELIVERY_RESULT_APPLIED);
    response->set_message("delivery settled");
    response->set_disposition(disposition);
    response->set_train_update_id(request.train_update_id());
    cv_.notify_all();
}

void SamplePoolCoordinator::Nack(
    const rl::training::v1::NackBatchReq& request,
    rl::training::v1::DeliveryRsp* response) {
    std::unique_lock<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_delivery_id(request.delivery_id());

    if (!IsServiceIdentityValid(request.consumer()) ||
        request.delivery_id().empty() ||
        !DeliveryBelongsToInstanceLocked(request.delivery_id())) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("delivery or consumer identity is invalid");
        return;
    }
    if (const DeliveryRecord* history =
            DeliveryHistoryLocked(request.delivery_id())) {
        response->set_result(
            rl::training::v1::DELIVERY_RESULT_ALREADY_APPLIED);
        response->set_message("delivery settlement already applied");
        return;
    }
    if (!has_lease_ || lease_.delivery_id != request.delivery_id()) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_NOT_FOUND);
        response->set_message("active delivery not found");
        return;
    }
    if (lease_.consumer_instance_id != ServiceKey(request.consumer())) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("delivery belongs to another consumer");
        return;
    }

    const int64_t affected = lease_.transition_count;
    backend_->RestoreReady(std::move(lease_.items));
    ready_transitions_ += affected;
    ready_estimated_bytes_ += lease_.estimated_bytes;
    ++nack_count_;
    redelivery_count_ += affected;
    DeliveryRecord record;
    record.result = rl::training::v1::DELIVERY_RESULT_APPLIED;
    RememberDeliveryLocked(request.delivery_id(), record);
    lease_ = Lease{};
    has_lease_ = false;

    response->set_result(rl::training::v1::DELIVERY_RESULT_APPLIED);
    response->set_message("delivery returned to READY");
    cv_.notify_all();
}

void SamplePoolCoordinator::RenewLease(
    const rl::training::v1::RenewLeaseReq& request,
    rl::training::v1::DeliveryRsp* response) {
    std::unique_lock<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_delivery_id(request.delivery_id());
    if (!IsServiceIdentityValid(request.consumer()) ||
        request.delivery_id().empty() || request.lease_timeout_ms() <= 0 ||
        !DeliveryBelongsToInstanceLocked(request.delivery_id())) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("lease renewal request is invalid");
        return;
    }
    if (!has_lease_ || lease_.delivery_id != request.delivery_id()) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_NOT_FOUND);
        response->set_message("active delivery not found");
        return;
    }
    if (lease_.consumer_instance_id != ServiceKey(request.consumer())) {
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("delivery belongs to another consumer");
        return;
    }

    lease_.deadline = std::chrono::steady_clock::now() +
                      std::chrono::milliseconds(request.lease_timeout_ms());
    lease_.deadline_unix_ms = NowMs() + request.lease_timeout_ms();
    ++lease_renew_count_;
    response->set_result(rl::training::v1::DELIVERY_RESULT_APPLIED);
    response->set_message("lease renewed");
    response->set_lease_deadline_unix_ms(lease_.deadline_unix_ms);
}

void SamplePoolCoordinator::FinalizeSamplePool(
    const rl::training::v1::FinalizeSamplePoolReq& request,
    rl::training::v1::FinalizeSamplePoolRsp* response) {
    std::unique_lock<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    const std::string consumer_key = ServiceKey(request.consumer());
    if (!IsServiceIdentityValid(request.consumer()) ||
        request.finalization_id().empty() ||
        request.expected_sample_pool().component() != "sample-pool" ||
        request.expected_sample_pool().instance_id() != instance_id_ ||
        request.expected_sample_pool().lifecycle_epoch() != 1) {
        response->set_result(
            rl::training::v1::SAMPLE_POOL_FINALIZE_RESULT_REJECTED_IDENTITY);
        response->set_message("finalization identity is invalid");
        FillFinalizeResponseLocked(response);
        return;
    }
    if (finalized_) {
        const bool same = request.finalization_id() == finalization_id_ &&
                          consumer_key == finalization_consumer_key_;
        response->set_result(
            same ? rl::training::v1::
                       SAMPLE_POOL_FINALIZE_RESULT_ALREADY_FINALIZED
                 : rl::training::v1::
                       SAMPLE_POOL_FINALIZE_RESULT_REJECTED_CONFLICT);
        response->set_message(
            same ? "SamplePool already finalized"
                 : "SamplePool finalized by another identity");
        FillFinalizeResponseLocked(response);
        return;
    }
    if (has_lease_) {
        response->set_result(
            rl::training::v1::
                SAMPLE_POOL_FINALIZE_RESULT_REJECTED_ACTIVE_LEASE);
        response->set_message("active lease must settle before finalization");
        FillFinalizeResponseLocked(response);
        return;
    }

    std::vector<StoredTransition> tail = backend_->ExtractAllReady();
    finalized_transition_count_ = tail.size();
    for (const auto& item : tail) RemoveResidentItemLocked(item);
    shutdown_untrained_transition_count_ += finalized_transition_count_;
    ready_transitions_ = 0;
    ready_estimated_bytes_ = 0;
    finalized_ = true;
    finalization_id_ = request.finalization_id();
    finalization_consumer_key_ = consumer_key;
    finalized_at_unix_ms_ = NowMs();

    response->set_result(
        rl::training::v1::SAMPLE_POOL_FINALIZE_RESULT_FINALIZED);
    response->set_message("READY tail settled as SHUTDOWN_UNTRAINED");
    FillFinalizeResponseLocked(response);
    cv_.notify_all();
}

void SamplePoolCoordinator::FillStatusScalarsLocked(
    rl::training::v1::SamplePoolStatusRsp* response) const {
    FillContractIdentity(response->mutable_contract());
    FillServiceIdentity(response->mutable_sample_pool());
    response->set_ready(true);
    response->set_push_attempt_count(push_attempt_count_);
    response->set_accepted_unique_transitions(
        accepted_unique_transitions_);
    response->set_accepted_unique_envelopes(
        accepted_unique_envelopes_);
    response->set_duplicate_push_attempt_count(
        duplicate_push_attempt_count_);
    response->set_duplicate_transition_attempts(
        duplicate_transition_attempts_);
    response->set_rejected_push_attempt_count(
        rejected_push_attempt_count_);
    response->set_rejected_transition_attempts(
        rejected_transition_attempts_);
    response->set_acked_unique_transitions(acked_unique_transitions_);
    response->set_acked_unique_deliveries(acked_unique_deliveries_);
    response->set_ready_transitions(ready_transitions_);
    response->set_leased_transitions(
        has_lease_ ? lease_.transition_count : 0);
    response->set_resident_transitions(resident_transitions_);
    response->set_resident_envelopes(resident_by_envelope_.size());
    response->set_resident_estimated_bytes(resident_estimated_bytes_);
    response->set_capacity_transitions(config_.capacity_transitions);
    response->set_capacity_bytes(config_.capacity_bytes);
    response->set_pressure_state(PressureStateLocked());
    response->set_redelivery_count(redelivery_count_);
    response->set_nack_count(nack_count_);
    response->set_expired_lease_count(expired_lease_count_);
    if (latest_ack_unix_ms_ > 0) {
        response->set_latest_ack_at_unix_ms(latest_ack_unix_ms_);
    }
    response->set_target_hit_count(target_hit_count_);
    response->set_draw_attempt_count(draw_attempt_count_);
    response->set_drawn_transition_slot_count(
        drawn_transition_slot_count_);
    response->set_empty_timeout_count(empty_timeout_count_);
    response->set_last_error(last_error_);
    response->set_trained_transition_count(trained_transition_count_);
    response->set_invalid_transition_count(invalid_transition_count_);
    response->set_shutdown_untrained_transition_count(
        shutdown_untrained_transition_count_);
    response->set_lease_renew_count(lease_renew_count_);
    response->set_backend_type(
        rl::training::v1::SAMPLE_BACKEND_TYPE_LOCAL_MEMORY);
    response->set_max_concurrent_consumers(1);
    response->set_active_consumer_count(has_lease_ ? 1 : 0);
    response->set_consumer_busy_count(consumer_busy_count_);
    response->set_ingress_ready(!finalized_);
    response->set_pool_ready(ready_transitions_ > 0);
    response->set_timestamp_unix_ms(NowMs());
    if (!backend_->ready().empty()) {
        const auto oldest = std::min_element(
            backend_->ready().begin(), backend_->ready().end(),
            [](const StoredTransition& left,
               const StoredTransition& right) {
                return left.insert_sequence < right.insert_sequence;
            });
        response->set_oldest_ready_transition_age_ms(
            NowMs() - oldest->inserted_at_unix_ms);
        uint64_t minimum_step = std::numeric_limits<uint64_t>::max();
        uint64_t maximum_step = 0;
        for (const auto& item : backend_->ready()) {
            minimum_step = std::min(
                minimum_step,
                item.transition.behavior_model_step());
            maximum_step = std::max(
                maximum_step,
                item.transition.behavior_model_step());
        }
        response->set_minimum_ready_model_step(minimum_step);
        response->set_maximum_ready_model_step(maximum_step);
    }
    response->set_evicted_transition_count(evicted_transition_count_);
    response->set_evicted_envelope_count(evicted_envelope_count_);
    response->set_unsampled_evicted_transition_count(
        unsampled_evicted_transition_count_);
    response->set_previously_drawn_evicted_transition_count(
        previously_drawn_evicted_transition_count_);
    response->set_finalized(finalized_);
    response->set_finalization_id(finalization_id_);
    if (finalized_at_unix_ms_ > 0) {
        response->set_finalized_at_unix_ms(finalized_at_unix_ms_);
    }
    response->set_finalized_transition_count(
        finalized_transition_count_);
}

void SamplePoolCoordinator::GetStatus(
    const rl::training::v1::SamplePoolStatusReq&,
    rl::training::v1::SamplePoolStatusRsp* response) {
    std::unique_lock<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    FillStatusScalarsLocked(response);
}

const std::string& SamplePoolCoordinator::instance_id() const {
    return instance_id_;
}
