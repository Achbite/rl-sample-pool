#include "store/sample_store.h"

#include <algorithm>
#include <cmath>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <iomanip>
#include <openssl/evp.h>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {

constexpr char kAIServerSampleProducerComponent[] = "aiserver";

bool FiniteRepeated(const google::protobuf::RepeatedField<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

}  // namespace

SampleStore::SampleStore(const DistributorConfig& config)
    : config_(config), instance_id_(CreateInstanceId("sample-pool")) {}

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
    out << prefix << "-" << NowMs() << "-" << std::hex << std::setw(16)
        << std::setfill('0') << rng();
    return out.str();
}

int64_t SampleStore::CountSamples(const rl::training::v1::SampleBatch& batch) {
    return static_cast<int64_t>(batch.samples_size());
}

int64_t SampleStore::EstimateBytes(const rl::training::v1::SampleBatch& batch) {
    return static_cast<int64_t>(batch.SpaceUsedLong());
}

std::string SampleStore::DeterministicSerialize(
    const rl::training::v1::SampleBatch& batch,
    bool clear_payload_digest) {
    rl::training::v1::SampleBatch copy(batch);
    if (clear_payload_digest) copy.clear_payload_digest();
    std::string serialized;
    google::protobuf::io::StringOutputStream output(&serialized);
    google::protobuf::io::CodedOutputStream coded(&output);
    coded.SetSerializationDeterministic(true);
    if (!copy.SerializeToCodedStream(&coded) || coded.HadError()) {
        throw std::runtime_error("failed to serialize SampleBatch");
    }
    coded.Trim();
    return serialized;
}

std::string SampleStore::Sha256Hex(const std::string& data) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr) throw std::runtime_error("EVP_MD_CTX_new failed");
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_size = 0;
    const bool ok = EVP_DigestInit_ex(context, EVP_sha256(), nullptr) == 1 &&
                    EVP_DigestUpdate(context, data.data(), data.size()) == 1 &&
                    EVP_DigestFinal_ex(context, digest, &digest_size) == 1;
    EVP_MD_CTX_free(context);
    if (!ok || digest_size != 32) {
        throw std::runtime_error("SHA-256 calculation failed");
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (unsigned int index = 0; index < digest_size; ++index) {
        output << std::setw(2) << static_cast<unsigned int>(digest[index]);
    }
    return output.str();
}

SampleStore::BatchFingerprint SampleStore::FingerprintBatch(
    const rl::training::v1::SampleBatch& batch) {
    const std::string serialized = DeterministicSerialize(batch, false);
    return BatchFingerprint{
        batch.payload_digest().hex(),
        static_cast<uint64_t>(serialized.size()),
    };
}

bool SampleStore::IsSha256(const rl::common::v1::ContentDigest& digest) {
    if (digest.algorithm() != rl::common::v1::DIGEST_ALGORITHM_SHA256 ||
        digest.hex().size() != 64) {
        return false;
    }
    return std::all_of(digest.hex().begin(), digest.hex().end(), [](char value) {
        return (value >= '0' && value <= '9') ||
               (value >= 'a' && value <= 'f');
    });
}

bool SampleStore::IsServiceIdentityValid(
    const rl::common::v1::ServiceInstanceIdentity& identity) {
    return !identity.component().empty() && !identity.instance_id().empty() &&
           identity.lifecycle_epoch() > 0;
}

std::string SampleStore::ServiceKey(
    const rl::common::v1::ServiceInstanceIdentity& identity) {
    return identity.component() + "\x1f" + identity.instance_id() + "\x1f" +
           std::to_string(identity.lifecycle_epoch());
}

bool SampleStore::IsModelIdentityValid(
    const rl::training::v1::ModelIdentity& identity) {
    return !identity.model_lineage_id().empty() &&
           IsSha256(identity.artifact_digest()) &&
           IsSha256(identity.manifest_digest());
}

bool SampleStore::IsSchemaIdentityValid(
    const rl::common::v1::SchemaIdentity& identity) {
    return !identity.schema_id().empty() && identity.schema_version() > 0 &&
           IsSha256(identity.canonical_digest());
}

std::string SampleStore::PolicyKey(
    const rl::training::v1::ModelIdentity& model,
    const rl::training::v1::TrainingSemanticsIdentity& semantics) {
    return model.model_lineage_id() + "\x1f" +
           std::to_string(model.model_version()) + "\x1f" +
           model.artifact_digest().hex() + "\x1f" +
           model.manifest_digest().hex() + "\x1f" +
           semantics.semantics_digest().hex();
}

bool SampleStore::ContractMatchesConfig(
    const rl::common::v1::ContractIdentity& contract) const {
    return contract.package_name() == config_.contract.package_name &&
           contract.package_version() == config_.contract.package_version &&
           IsSha256(contract.source_digest()) &&
           contract.source_digest().hex() == config_.contract.source_digest &&
           IsSha256(contract.artifact_digest()) &&
           contract.artifact_digest().hex() == config_.contract.artifact_digest &&
           contract.platform() == config_.contract.platform &&
           contract.generator_identity() == config_.contract.generator_identity;
}

bool SampleStore::ValidateSemantics(
    const rl::training::v1::TrainingSemanticsIdentity& semantics,
    std::string* error) const {
    if (semantics.training_contract_id().empty() ||
        !IsSchemaIdentityValid(semantics.observation_schema()) ||
        !IsSchemaIdentityValid(semantics.action_schema()) ||
        !IsSchemaIdentityValid(semantics.reward_schema()) ||
        semantics.policy_distribution_schema_id().empty() ||
        semantics.model_architecture_id().empty() ||
        !IsSha256(semantics.semantics_digest())) {
        *error = "training semantics identity is incomplete";
        return false;
    }
    return true;
}

bool SampleStore::ValidateBatchLocked(
    const rl::training::v1::SampleBatch& batch,
    std::string* error) const {
    if (batch.batch_id().empty() || batch.actor_session_id().empty() ||
        batch.trajectory_id().empty() || batch.created_at_unix_ms() <= 0) {
        *error = "batch identity is incomplete";
        return false;
    }
    if (!IsServiceIdentityValid(batch.producer()) ||
        batch.producer().component() !=
            kAIServerSampleProducerComponent) {
        *error = "producer must be an aiserver service identity";
        return false;
    }
    if (!ContractMatchesConfig(batch.contract())) {
        *error = "contract identity does not match the configured artifact";
        return false;
    }
    if (!IsModelIdentityValid(batch.behavior_policy().model()) ||
        batch.behavior_policy().distribution_schema_id().empty() ||
        !IsSha256(batch.behavior_policy().policy_spec_digest())) {
        *error = "behavior policy identity is invalid";
        return false;
    }
    if (!ValidateSemantics(batch.training_semantics(), error)) return false;
    if (batch.behavior_policy().distribution_schema_id() !=
        batch.training_semantics().policy_distribution_schema_id()) {
        *error = "behavior policy and training semantics disagree";
        return false;
    }
    if (!IsSha256(batch.payload_digest())) {
        *error = "payload digest is not a canonical SHA-256 value";
        return false;
    }
    try {
        if (Sha256Hex(DeterministicSerialize(batch, true)) !=
            batch.payload_digest().hex()) {
            *error = "payload digest does not match deterministic bytes";
            return false;
        }
    } catch (const std::exception& exception) {
        *error = exception.what();
        return false;
    }
    if (batch.samples_size() == 0) {
        *error = "empty fragment is invalid";
        return false;
    }
    if (batch.last_action_step() < batch.first_action_step() ||
        batch.last_action_step() - batch.first_action_step() + 1 !=
            static_cast<uint64_t>(batch.samples_size())) {
        *error = "fragment action-step range does not match sample count";
        return false;
    }

    for (int index = 0; index < batch.samples_size(); ++index) {
        const auto& sample = batch.samples(index);
        const uint64_t expected_step = batch.first_action_step() + index;
        if (sample.action_step() != expected_step) {
            *error = "sample action_step is not contiguous";
            return false;
        }
        if (sample.observation_size() == 0 ||
            sample.observation_size() != sample.next_observation_size() ||
            !FiniteRepeated(sample.observation()) ||
            !FiniteRepeated(sample.next_observation()) || sample.action() < 0 ||
            !std::isfinite(sample.reward()) ||
            !std::isfinite(sample.old_log_probability()) ||
            !std::isfinite(sample.old_value_prediction()) ||
            (sample.terminated() && sample.truncated())) {
            *error = "sample payload is invalid";
            return false;
        }
        switch (sample.end_kind()) {
            case rl::training::v1::TRANSITION_END_KIND_CONTINUING:
                if (sample.terminated() || sample.truncated()) {
                    *error = "continuing transition carries an end flag";
                    return false;
                }
                break;
            case rl::training::v1::TRANSITION_END_KIND_ENVIRONMENT_TERMINATED:
                if (!sample.terminated() || sample.truncated()) {
                    *error = "terminated transition flags are inconsistent";
                    return false;
                }
                break;
            case rl::training::v1::TRANSITION_END_KIND_EXTERNAL_TRUNCATION:
                if (sample.terminated() || !sample.truncated()) {
                    *error = "truncated transition flags are inconsistent";
                    return false;
                }
                break;
            case rl::training::v1::TRANSITION_END_KIND_PRODUCER_ABORT:
                *error = "producer-aborted fragments cannot enter training";
                return false;
            default:
                *error = "transition end kind is unspecified";
                return false;
        }
        if (index + 1 < batch.samples_size() &&
            (sample.terminated() || sample.truncated())) {
            *error = "only the final sample may end a fragment";
            return false;
        }
    }

    const auto& final_sample = batch.samples(batch.samples_size() - 1);
    const bool final_ends = final_sample.terminated() || final_sample.truncated();
    if (batch.trajectory_end() != final_ends) {
        *error = "trajectory_end does not match the final transition";
        return false;
    }
    if (final_sample.terminated()) {
        if (batch.bootstrap_valid() || std::fabs(batch.bootstrap_value()) > 1e-6f) {
            *error = "terminated fragment must not bootstrap";
            return false;
        }
    } else if (!batch.bootstrap_valid() ||
               !std::isfinite(batch.bootstrap_value())) {
        *error = "continuing or truncated fragment requires finite bootstrap";
        return false;
    }
    return true;
}

bool SampleStore::DeliveryBelongsToInstanceLocked(
    const std::string& delivery_id) const {
    return delivery_id.rfind(instance_id_ + "-delivery-", 0) == 0;
}

bool SampleStore::CapacityAllowsLocked(int64_t samples,
                                       int64_t fragments,
                                       int64_t estimated_bytes) const {
    return resident_samples_ + samples <= config_.max_queue_samples &&
           resident_fragments_ + fragments <= config_.max_queue_fragments &&
           resident_estimated_bytes_ + estimated_bytes <=
               config_.max_queue_estimated_bytes;
}

rl::training::v1::PressureState SampleStore::PressureStateLocked() const {
    const double sample_ratio =
        static_cast<double>(resident_samples_) / config_.max_queue_samples;
    const double fragment_ratio =
        static_cast<double>(resident_fragments_) / config_.max_queue_fragments;
    const double byte_ratio = static_cast<double>(resident_estimated_bytes_) /
                              config_.max_queue_estimated_bytes;
    const double ratio = std::max({sample_ratio, fragment_ratio, byte_ratio});
    if (ratio >= 1.0) return rl::training::v1::PRESSURE_STATE_FULL;
    if (ratio >= config_.high_watermark_ratio) {
        return rl::training::v1::PRESSURE_STATE_HIGH;
    }
    return rl::training::v1::PRESSURE_STATE_NORMAL;
}

void SampleStore::FillServiceIdentity(
    rl::common::v1::ServiceInstanceIdentity* identity) const {
    identity->set_component("sample-pool");
    identity->set_instance_id(instance_id_);
    identity->set_lifecycle_epoch(1);
}

void SampleStore::FillContractIdentity(
    rl::common::v1::ContractIdentity* identity) const {
    identity->set_package_name(config_.contract.package_name);
    identity->set_package_version(config_.contract.package_version);
    identity->mutable_source_digest()->set_algorithm(
        rl::common::v1::DIGEST_ALGORITHM_SHA256);
    identity->mutable_source_digest()->set_hex(config_.contract.source_digest);
    identity->mutable_artifact_digest()->set_algorithm(
        rl::common::v1::DIGEST_ALGORITHM_SHA256);
    identity->mutable_artifact_digest()->set_hex(config_.contract.artifact_digest);
    identity->set_platform(config_.contract.platform);
    identity->set_generator_identity(config_.contract.generator_identity);
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

void SampleStore::RememberCompletedBatchLocked(
    const std::string& batch_id,
    const BatchFingerprint& fingerprint) {
    if (completed_batch_fingerprints_.find(batch_id) ==
        completed_batch_fingerprints_.end()) {
        completed_batch_order_.push_back(batch_id);
    }
    completed_batch_fingerprints_[batch_id] = fingerprint;
    while (static_cast<int64_t>(completed_batch_order_.size()) >
           config_.max_dedup_entries) {
        const std::string oldest = completed_batch_order_.front();
        completed_batch_order_.pop_front();
        completed_batch_fingerprints_.erase(oldest);
    }
}

int64_t SampleStore::ReadySamplesForPolicyLocked(
    const std::string& policy_key) const {
    const auto found = policy_counters_.find(policy_key);
    return found == policy_counters_.end() ? 0 : found->second.ready_samples;
}

void SampleStore::RequeueLeaseLocked(bool expired) {
    if (!has_lease_) return;
    auto& queue = ready_by_policy_[lease_.policy_key];
    for (auto it = lease_.batches.rbegin(); it != lease_.batches.rend(); ++it) {
        queue.push_front(std::move(*it));
    }
    ready_samples_ += lease_.sample_count;
    ready_fragments_ += static_cast<int64_t>(lease_.batches.size());
    ready_estimated_bytes_ += lease_.estimated_bytes;
    auto& counters = policy_counters_[lease_.policy_key];
    counters.ready_samples += lease_.sample_count;
    counters.ready_fragments += static_cast<int64_t>(lease_.batches.size());
    counters.leased_samples -= lease_.sample_count;
    counters.leased_fragments -= static_cast<int64_t>(lease_.batches.size());

    DeliveryRecord record;
    if (expired) {
        ++expired_lease_count_;
        record.result = rl::training::v1::DELIVERY_RESULT_EXPIRED;
    } else {
        ++nack_count_;
        record.result = rl::training::v1::DELIVERY_RESULT_APPLIED;
    }
    ++redelivery_count_;
    RememberDeliveryLocked(lease_.delivery_id, record);
    lease_ = Lease{};
    has_lease_ = false;
    cv_.notify_all();
}

void SampleStore::ReclaimExpiredLeaseLocked() {
    if (has_lease_ && std::chrono::steady_clock::now() >= lease_.deadline) {
        RequeueLeaseLocked(true);
    }
}

void SampleStore::Push(const rl::training::v1::SampleBatch& batch,
                       rl::training::v1::PushSamplesRsp* response) {
    const int64_t sample_count = CountSamples(batch);
    const int64_t estimated_bytes = EstimateBytes(batch);
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    ++push_attempt_count_;
    response->set_batch_id(batch.batch_id());
    FillServiceIdentity(response->mutable_distributor());

    std::string error;
    if (!ValidateBatchLocked(batch, &error)) {
        ++rejected_push_attempt_count_;
        rejected_sample_attempts_ += sample_count;
        last_error_ = error;
        response->set_ret_code(-1);
        response->set_message(error);
        response->set_result(
            error.find("identity") != std::string::npos ||
                    error.find("contract") != std::string::npos
                ? rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY
                : rl::training::v1::PUSH_RESULT_REJECTED_INVALID);
    } else {
        const BatchFingerprint fingerprint = FingerprintBatch(batch);
        const auto active = active_batch_fingerprints_.find(batch.batch_id());
        const auto completed = completed_batch_fingerprints_.find(batch.batch_id());
        if (active != active_batch_fingerprints_.end() ||
            completed != completed_batch_fingerprints_.end()) {
            const BatchFingerprint& prior =
                active != active_batch_fingerprints_.end() ? active->second
                                                            : completed->second;
            if (prior == fingerprint) {
                ++duplicate_push_attempt_count_;
                duplicate_sample_attempts_ += sample_count;
                response->set_ret_code(0);
                response->set_message("batch already accepted");
                response->set_result(rl::training::v1::PUSH_RESULT_DUPLICATE);
            } else {
                ++rejected_push_attempt_count_;
                rejected_sample_attempts_ += sample_count;
                last_error_ = "batch_id conflicts with a prior payload";
                response->set_ret_code(-1);
                response->set_message(last_error_);
                response->set_result(
                    rl::training::v1::PUSH_RESULT_REJECTED_INVALID);
            }
        } else if (!CapacityAllowsLocked(sample_count, 1, estimated_bytes)) {
            ++rejected_push_attempt_count_;
            rejected_sample_attempts_ += sample_count;
            last_error_ = "queue or dedup capacity exceeded";
            response->set_ret_code(-1);
            response->set_message(last_error_);
            response->set_result(rl::training::v1::PUSH_RESULT_REJECTED_CAPACITY);
        } else {
            const std::string policy_key =
                PolicyKey(batch.behavior_policy().model(),
                          batch.training_semantics());
            const auto known_policy = behavior_policy_by_key_.find(policy_key);
            if (known_policy != behavior_policy_by_key_.end() &&
                known_policy->second.SerializeAsString() !=
                    batch.behavior_policy().SerializeAsString()) {
                ++rejected_push_attempt_count_;
                rejected_sample_attempts_ += sample_count;
                last_error_ = "policy key conflicts with behavior identity";
                response->set_ret_code(-1);
                response->set_message(last_error_);
                response->set_result(
                    rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY);
            } else {
                behavior_policy_by_key_[policy_key] = batch.behavior_policy();
                StoredBatch stored;
                stored.batch = batch;
                stored.fingerprint = fingerprint;
                stored.policy_key = policy_key;
                stored.sample_count = sample_count;
                stored.estimated_bytes = estimated_bytes;
                ready_by_policy_[policy_key].push_back(std::move(stored));
                auto& counters = policy_counters_[policy_key];
                counters.behavior_model = batch.behavior_policy().model();
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
                active_batch_fingerprints_[batch.batch_id()] = fingerprint;
                latest_push_unix_ms_ = NowMs();
                last_error_.clear();
                response->set_ret_code(0);
                response->set_message("accepted");
                response->set_result(rl::training::v1::PUSH_RESULT_ACCEPTED);
                response->set_accepted_samples(sample_count);
                response->set_accepted_unique_samples(sample_count);
            }
        }
    }

    response->set_queue_size(ready_samples_);
    response->set_resident_samples(resident_samples_);
    response->set_resident_fragments(resident_fragments_);
    response->set_resident_estimated_bytes(resident_estimated_bytes_);
    response->set_pressure_state(PressureStateLocked());
    cv_.notify_all();
}

void SampleStore::GetBatch(
    const rl::training::v1::GetBatchReq& request,
    rl::training::v1::GetBatchRsp* response,
    const std::function<bool()>& is_cancelled) {
    const int target_samples = request.batch_size();
    const int timeout_ms = request.timeout_ms() > 0
                               ? request.timeout_ms()
                               : config_.default_get_timeout_ms;
    const int lease_timeout_ms = request.lease_timeout_ms() > 0
                                     ? request.lease_timeout_ms()
                                     : config_.default_lease_timeout_ms;
    const auto selection = request.selection_policy();
    const auto start = std::chrono::steady_clock::now();
    const auto wait_deadline = start + std::chrono::milliseconds(timeout_ms);
    const std::string policy_key =
        PolicyKey(request.target_model(), request.required_semantics());

    std::unique_lock<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    FillServiceIdentity(response->mutable_distributor());
    std::string semantics_error;
    if (!IsServiceIdentityValid(request.consumer()) || target_samples <= 0 ||
        timeout_ms <= 0 || lease_timeout_ms <= 0 ||
        !IsModelIdentityValid(request.target_model()) ||
        !ValidateSemantics(request.required_semantics(), &semantics_error) ||
        (selection != rl::training::v1::BATCH_SELECTION_POLICY_TARGET_ONLY &&
         selection !=
             rl::training::v1::BATCH_SELECTION_POLICY_DRAIN_AVAILABLE)) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::GET_BATCH_RESULT_REJECTED);
        response->set_message("invalid GetBatch request");
        response->set_queue_size(ready_samples_);
        return;
    }
    if (has_lease_) {
        ++consumer_busy_count_;
        response->set_ret_code(2);
        response->set_result(rl::training::v1::GET_BATCH_RESULT_BUSY);
        response->set_message("another delivery is still leased");
        response->set_queue_size(ready_samples_);
        response->set_leased_samples(lease_.sample_count);
        return;
    }

    const auto ready_for_selection = [&]() {
        const int64_t available = ReadySamplesForPolicyLocked(policy_key);
        return selection ==
                       rl::training::v1::BATCH_SELECTION_POLICY_TARGET_ONLY
                   ? available >= target_samples
                   : available > 0;
    };
    while (!ready_for_selection() &&
           std::chrono::steady_clock::now() < wait_deadline) {
        if (is_cancelled && is_cancelled()) {
            response->set_ret_code(-1);
            response->set_result(rl::training::v1::GET_BATCH_RESULT_REJECTED);
            response->set_message("request cancelled");
            response->set_queue_size(ready_samples_);
            return;
        }
        cv_.wait_until(lock, std::min(
                                wait_deadline,
                                std::chrono::steady_clock::now() +
                                    std::chrono::milliseconds(50)));
        ReclaimExpiredLeaseLocked();
        if (has_lease_) {
            ++consumer_busy_count_;
            response->set_ret_code(2);
            response->set_result(rl::training::v1::GET_BATCH_RESULT_BUSY);
            response->set_message("another delivery is still leased");
            response->set_queue_size(ready_samples_);
            response->set_leased_samples(lease_.sample_count);
            return;
        }
    }
    if (!ready_for_selection()) {
        ++empty_timeout_count_;
        response->set_ret_code(1);
        response->set_result(rl::training::v1::GET_BATCH_RESULT_TIMEOUT);
        response->set_message("matching policy not available before deadline");
        response->set_queue_size(ready_samples_);
        response->set_wait_ms(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
                .count());
        return;
    }

    Lease new_lease;
    new_lease.consumer_instance_id = ServiceKey(request.consumer());
    new_lease.policy_key = policy_key;
    new_lease.behavior_policy = behavior_policy_by_key_.at(policy_key);
    new_lease.delivery_id =
        instance_id_ + "-delivery-" + std::to_string(next_delivery_seq_++);
    new_lease.deadline = std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(lease_timeout_ms);
    new_lease.deadline_unix_ms = NowMs() + lease_timeout_ms;

    auto& queue = ready_by_policy_[policy_key];
    while (!queue.empty() &&
           (new_lease.sample_count < target_samples ||
            new_lease.batches.empty())) {
        StoredBatch stored = std::move(queue.front());
        queue.pop_front();
        ready_samples_ -= stored.sample_count;
        --ready_fragments_;
        ready_estimated_bytes_ -= stored.estimated_bytes;
        auto& counters = policy_counters_[policy_key];
        counters.ready_samples -= stored.sample_count;
        --counters.ready_fragments;
        new_lease.sample_count += stored.sample_count;
        new_lease.estimated_bytes += stored.estimated_bytes;
        new_lease.batches.push_back(std::move(stored));
    }
    auto& counters = policy_counters_[policy_key];
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
    response->set_result(rl::training::v1::GET_BATCH_RESULT_LEASED);
    response->set_message("leased");
    response->set_delivery_id(new_lease.delivery_id);
    response->set_lease_deadline_unix_ms(new_lease.deadline_unix_ms);
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
    *response->mutable_behavior_policy() = new_lease.behavior_policy;
    lease_ = std::move(new_lease);
    has_lease_ = true;
    latest_consume_unix_ms_ = NowMs();
}

void SampleStore::FillDeliveryResponseLocked(
    rl::training::v1::DeliveryRsp* response) const {
    response->set_queue_size(ready_samples_);
    if (has_lease_) {
        response->set_lease_deadline_unix_ms(lease_.deadline_unix_ms);
    }
}

void SampleStore::Ack(const rl::training::v1::AckBatchReq& request,
                      rl::training::v1::DeliveryRsp* response) {
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_delivery_id(request.delivery_id());
    FillDeliveryResponseLocked(response);
    const bool trained =
        request.disposition() == rl::training::v1::ACK_DISPOSITION_TRAINED;
    if (!IsServiceIdentityValid(request.consumer()) ||
        request.delivery_id().empty() ||
        request.disposition() ==
            rl::training::v1::ACK_DISPOSITION_UNSPECIFIED ||
        (trained && request.train_update_id().empty())) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("invalid AckBatch request");
        return;
    }
    if (!DeliveryBelongsToInstanceLocked(request.delivery_id())) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("delivery belongs to another pool instance");
        return;
    }
    const DeliveryRecord* history = DeliveryHistoryLocked(request.delivery_id());
    if (history != nullptr) {
        if (history->result == rl::training::v1::DELIVERY_RESULT_EXPIRED) {
            response->set_ret_code(1);
            response->set_result(rl::training::v1::DELIVERY_RESULT_EXPIRED);
            response->set_message("delivery lease expired");
            return;
        }
        if (history->disposition != request.disposition() ||
            history->train_update_id != request.train_update_id()) {
            response->set_ret_code(-1);
            response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
            response->set_message("Ack retry conflicts with applied result");
            return;
        }
        response->set_ret_code(0);
        response->set_result(
            rl::training::v1::DELIVERY_RESULT_ALREADY_APPLIED);
        response->set_message("delivery already completed");
        response->set_disposition(history->disposition);
        response->set_train_update_id(history->train_update_id);
        return;
    }
    if (!has_lease_ || lease_.delivery_id != request.delivery_id()) {
        response->set_ret_code(1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_NOT_FOUND);
        response->set_message("delivery not found");
        return;
    }
    if (lease_.consumer_instance_id != ServiceKey(request.consumer())) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("consumer identity does not own delivery");
        return;
    }

    const int64_t samples = lease_.sample_count;
    const int64_t fragments = static_cast<int64_t>(lease_.batches.size());
    auto& counters = policy_counters_[lease_.policy_key];
    counters.leased_samples -= samples;
    counters.leased_fragments -= fragments;
    counters.acked_samples += samples;
    counters.acked_fragments += fragments;
    switch (request.disposition()) {
        case rl::training::v1::ACK_DISPOSITION_TRAINED:
            counters.trained_samples += samples;
            trained_sample_count_ += samples;
            break;
        case rl::training::v1::ACK_DISPOSITION_STALE:
            counters.stale_samples += samples;
            stale_sample_count_ += samples;
            break;
        case rl::training::v1::ACK_DISPOSITION_INVALID:
            counters.invalid_samples += samples;
            invalid_sample_count_ += samples;
            break;
        case rl::training::v1::ACK_DISPOSITION_SHUTDOWN_UNTRAINED:
            counters.shutdown_untrained_samples += samples;
            shutdown_untrained_sample_count_ += samples;
            break;
        default:
            break;
    }
    resident_samples_ -= samples;
    resident_fragments_ -= fragments;
    resident_estimated_bytes_ -= lease_.estimated_bytes;
    acked_unique_samples_ += samples;
    acked_unique_batches_ += fragments;
    for (const auto& stored : lease_.batches) {
        const auto active =
            active_batch_fingerprints_.find(stored.batch.batch_id());
        if (active != active_batch_fingerprints_.end()) {
            RememberCompletedBatchLocked(stored.batch.batch_id(), active->second);
            active_batch_fingerprints_.erase(active);
        }
    }
    latest_ack_unix_ms_ = NowMs();
    latest_consume_unix_ms_ = latest_ack_unix_ms_;
    DeliveryRecord record;
    record.result = rl::training::v1::DELIVERY_RESULT_APPLIED;
    record.disposition = request.disposition();
    record.train_update_id = request.train_update_id();
    RememberDeliveryLocked(lease_.delivery_id, record);
    lease_ = Lease{};
    has_lease_ = false;
    response->set_ret_code(0);
    response->set_result(rl::training::v1::DELIVERY_RESULT_APPLIED);
    response->set_message("acked");
    response->set_affected_samples(samples);
    response->set_disposition(record.disposition);
    response->set_train_update_id(record.train_update_id);
    response->set_queue_size(ready_samples_);
    cv_.notify_all();
}

void SampleStore::Nack(const rl::training::v1::NackBatchReq& request,
                       rl::training::v1::DeliveryRsp* response) {
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_delivery_id(request.delivery_id());
    FillDeliveryResponseLocked(response);
    if (!IsServiceIdentityValid(request.consumer()) ||
        request.delivery_id().empty()) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("invalid NackBatch request");
        return;
    }
    if (!DeliveryBelongsToInstanceLocked(request.delivery_id())) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("delivery belongs to another pool instance");
        return;
    }
    const DeliveryRecord* history = DeliveryHistoryLocked(request.delivery_id());
    if (history != nullptr) {
        response->set_ret_code(
            history->result == rl::training::v1::DELIVERY_RESULT_EXPIRED ? 1 : 0);
        response->set_result(
            history->result == rl::training::v1::DELIVERY_RESULT_EXPIRED
                ? rl::training::v1::DELIVERY_RESULT_EXPIRED
                : rl::training::v1::DELIVERY_RESULT_ALREADY_APPLIED);
        response->set_message("delivery already completed");
        return;
    }
    if (!has_lease_ || lease_.delivery_id != request.delivery_id()) {
        response->set_ret_code(1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_NOT_FOUND);
        response->set_message("delivery not found");
        return;
    }
    if (lease_.consumer_instance_id != ServiceKey(request.consumer())) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("consumer identity does not own delivery");
        return;
    }
    const int64_t affected = lease_.sample_count;
    RequeueLeaseLocked(false);
    response->set_ret_code(0);
    response->set_result(rl::training::v1::DELIVERY_RESULT_APPLIED);
    response->set_message("nacked and requeued");
    response->set_affected_samples(affected);
    response->set_queue_size(ready_samples_);
}

void SampleStore::RenewLease(const rl::training::v1::RenewLeaseReq& request,
                             rl::training::v1::DeliveryRsp* response) {
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_delivery_id(request.delivery_id());
    FillDeliveryResponseLocked(response);
    if (!IsServiceIdentityValid(request.consumer()) ||
        request.delivery_id().empty() || request.lease_timeout_ms() <= 0) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("invalid RenewLease request");
        return;
    }
    if (!DeliveryBelongsToInstanceLocked(request.delivery_id())) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("delivery belongs to another pool instance");
        return;
    }
    const DeliveryRecord* history = DeliveryHistoryLocked(request.delivery_id());
    if (history != nullptr) {
        response->set_ret_code(1);
        response->set_result(history->result ==
                                     rl::training::v1::DELIVERY_RESULT_EXPIRED
                                 ? rl::training::v1::DELIVERY_RESULT_EXPIRED
                                 : rl::training::v1::DELIVERY_RESULT_NOT_FOUND);
        response->set_message("delivery is no longer active");
        return;
    }
    if (!has_lease_ || lease_.delivery_id != request.delivery_id()) {
        response->set_ret_code(1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_NOT_FOUND);
        response->set_message("delivery not found");
        return;
    }
    if (lease_.consumer_instance_id != ServiceKey(request.consumer())) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::DELIVERY_RESULT_REJECTED);
        response->set_message("consumer identity does not own delivery");
        return;
    }
    lease_.deadline = std::chrono::steady_clock::now() +
                      std::chrono::milliseconds(request.lease_timeout_ms());
    lease_.deadline_unix_ms = NowMs() + request.lease_timeout_ms();
    ++lease_renew_count_;
    response->set_ret_code(0);
    response->set_result(rl::training::v1::DELIVERY_RESULT_APPLIED);
    response->set_message("lease renewed");
    response->set_affected_samples(lease_.sample_count);
    response->set_lease_deadline_unix_ms(lease_.deadline_unix_ms);
}

void SampleStore::FillStatusLocked(
    rl::training::v1::DistributorStatusRsp* response) const {
    FillContractIdentity(response->mutable_contract());
    FillServiceIdentity(response->mutable_distributor());
    response->set_ready(true);
    response->set_push_attempt_count(push_attempt_count_);
    response->set_accepted_unique_samples(accepted_unique_samples_);
    response->set_accepted_unique_batches(accepted_unique_batches_);
    response->set_duplicate_push_attempt_count(duplicate_push_attempt_count_);
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
    response->set_capacity_estimated_bytes(config_.max_queue_estimated_bytes);
    response->set_pressure_state(PressureStateLocked());
    response->set_redelivery_count(redelivery_count_);
    response->set_nack_count(nack_count_);
    response->set_expired_lease_count(expired_lease_count_);
    response->set_latest_ack_at_unix_ms(latest_ack_unix_ms_);
    response->set_target_hit_count(target_hit_count_);
    response->set_partial_get_count(partial_get_count_);
    response->set_empty_timeout_count(empty_timeout_count_);
    response->set_last_error(last_error_);
    for (const auto& [key, counters] : policy_counters_) {
        (void)key;
        auto* status = response->add_behavior_versions();
        *status->mutable_behavior_model() = counters.behavior_model;
        status->set_ready_samples(counters.ready_samples);
        status->set_ready_fragments(counters.ready_fragments);
        status->set_leased_samples(counters.leased_samples);
        status->set_leased_fragments(counters.leased_fragments);
        status->set_acked_samples(counters.acked_samples);
        status->set_acked_fragments(counters.acked_fragments);
        status->set_trained_samples(counters.trained_samples);
        status->set_stale_samples(counters.stale_samples);
        status->set_invalid_samples(counters.invalid_samples);
        status->set_shutdown_untrained_samples(
            counters.shutdown_untrained_samples);
    }
    response->set_trained_sample_count(trained_sample_count_);
    response->set_stale_sample_count(stale_sample_count_);
    response->set_invalid_sample_count(invalid_sample_count_);
    response->set_shutdown_untrained_sample_count(
        shutdown_untrained_sample_count_);
    response->set_lease_renew_count(lease_renew_count_);
    response->set_backend_type(
        rl::training::v1::SAMPLE_BACKEND_TYPE_LOCAL_MEMORY);
    response->set_max_concurrent_consumers(1);
    response->set_active_consumer_count(has_lease_ ? 1 : 0);
    response->set_consumer_busy_count(consumer_busy_count_);
    response->set_ingress_ready(true);
    response->set_pool_ready(true);
    response->set_timestamp_unix_ms(NowMs());
}

void SampleStore::GetStatus(
    const rl::training::v1::DistributorStatusReq&,
    rl::training::v1::DistributorStatusRsp* response) {
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    FillStatusLocked(response);
}

const std::string& SampleStore::instance_id() const {
    return instance_id_;
}
