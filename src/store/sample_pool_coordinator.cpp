#include "store/sample_pool_coordinator.h"

#include <algorithm>
#include <cmath>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <iomanip>
#include <limits>
#include <openssl/evp.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

constexpr char kAIServerSampleProducerComponent[] = "aiserver";

bool FiniteRepeated(const google::protobuf::RepeatedField<float>& values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value);
    });
}

}  // namespace

SamplePoolCoordinator::SamplePoolCoordinator(const SamplePoolConfig& config)
    : config_(config),
      instance_id_(CreateInstanceId("sample-pool")),
      backend_(std::make_unique<LocalFragmentStore>()) {}

int64_t SamplePoolCoordinator::NowMs() {
    const auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               now.time_since_epoch())
        .count();
}

std::string SamplePoolCoordinator::CreateInstanceId(
    const std::string& prefix) {
    std::random_device random_device;
    std::mt19937_64 random(random_device());
    std::ostringstream output;
    output << prefix << "-" << NowMs() << "-" << std::hex << std::setw(16)
           << std::setfill('0') << random();
    return output.str();
}

int64_t SamplePoolCoordinator::CountSamples(
    const rl::training::v1::SampleBatch& batch) {
    return static_cast<int64_t>(batch.samples_size());
}

int64_t SamplePoolCoordinator::EstimateBytes(
    const rl::training::v1::SampleBatch& batch) {
    return static_cast<int64_t>(batch.ByteSizeLong());
}

std::string SamplePoolCoordinator::DeterministicSerialize(
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

std::string SamplePoolCoordinator::Sha256Hex(const std::string& data) {
    EVP_MD_CTX* context = EVP_MD_CTX_new();
    if (context == nullptr) {
        throw std::runtime_error("EVP_MD_CTX_new failed");
    }
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

SampleBatchFingerprint SamplePoolCoordinator::FingerprintBatch(
    const rl::training::v1::SampleBatch& batch) {
    return SampleBatchFingerprint{
        batch.payload_digest().hex(),
        static_cast<uint64_t>(DeterministicSerialize(batch, false).size()),
    };
}

bool SamplePoolCoordinator::IsSha256(
    const rl::common::v1::ContentDigest& digest) {
    if (digest.algorithm() != rl::common::v1::DIGEST_ALGORITHM_SHA256 ||
        digest.hex().size() != 64) {
        return false;
    }
    return std::all_of(digest.hex().begin(), digest.hex().end(), [](char value) {
        return (value >= '0' && value <= '9') ||
               (value >= 'a' && value <= 'f');
    });
}

bool SamplePoolCoordinator::IsServiceIdentityValid(
    const rl::common::v1::ServiceInstanceIdentity& identity) {
    return !identity.component().empty() && !identity.instance_id().empty() &&
           identity.lifecycle_epoch() > 0;
}

std::string SamplePoolCoordinator::ServiceKey(
    const rl::common::v1::ServiceInstanceIdentity& identity) {
    return identity.component() + "\x1f" + identity.instance_id() + "\x1f" +
           std::to_string(identity.lifecycle_epoch());
}

bool SamplePoolCoordinator::IsBehaviorPolicyReferenceValid(
    const rl::training::v1::BehaviorPolicyReference& identity) {
    return !identity.model_lineage_id().empty() && identity.has_model_step() &&
           !identity.distribution_schema_id().empty() &&
           IsSha256(identity.policy_spec_digest());
}

bool SamplePoolCoordinator::IsSchemaIdentityValid(
    const rl::common::v1::SchemaIdentity& identity) {
    return !identity.schema_id().empty() && identity.schema_version() > 0 &&
           IsSha256(identity.canonical_digest());
}

std::string SamplePoolCoordinator::PolicyKey(
    const rl::training::v1::BehaviorPolicyReference& policy,
    const rl::training::v1::TrainingSemanticsIdentity& semantics) {
    return policy.model_lineage_id() + "\x1f" +
           std::to_string(policy.model_step()) + "\x1f" +
           policy.distribution_schema_id() + "\x1f" +
           policy.policy_spec_digest().hex() + "\x1f" +
           semantics.semantics_digest().hex();
}

bool SamplePoolCoordinator::ContractMatchesConfig(
    const rl::common::v1::ContractIdentity& contract) const {
    return contract.package_name() == config_.contract.package_name &&
           contract.package_version() == config_.contract.package_version &&
           IsSha256(contract.source_digest()) &&
           contract.source_digest().hex() == config_.contract.source_digest &&
           IsSha256(contract.artifact_digest()) &&
           contract.artifact_digest().hex() ==
               config_.contract.artifact_digest &&
           contract.platform() == config_.contract.platform &&
           contract.generator_identity() ==
               config_.contract.generator_identity;
}

bool SamplePoolCoordinator::ValidateSemantics(
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

bool SamplePoolCoordinator::ValidateBatchLocked(
    const rl::training::v1::SampleBatch& batch,
    std::string* error,
    rl::training::v1::PushResult* rejection) const {
    *rejection = rl::training::v1::PUSH_RESULT_REJECTED_INVALID;
    if (batch.batch_id().empty() || batch.actor_session_id().empty() ||
        batch.trajectory_id().empty() || batch.created_at_unix_ms() <= 0) {
        *error = "batch identity is incomplete";
        return false;
    }
    if (!IsServiceIdentityValid(batch.producer()) ||
        batch.producer().component() != kAIServerSampleProducerComponent) {
        *error = "producer must be an aiserver service identity";
        *rejection = rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY;
        return false;
    }
    if (!ContractMatchesConfig(batch.contract())) {
        *error = "contract identity does not match the configured artifact";
        *rejection = rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY;
        return false;
    }
    if (!IsBehaviorPolicyReferenceValid(batch.behavior_policy())) {
        *error = "behavior policy reference is invalid";
        *rejection = rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY;
        return false;
    }
    if (!ValidateSemantics(batch.training_semantics(), error)) {
        *rejection = rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY;
        return false;
    }
    if (batch.behavior_policy().distribution_schema_id() !=
        batch.training_semantics().policy_distribution_schema_id()) {
        *error = "behavior policy and training semantics disagree";
        *rejection = rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY;
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
    if (batch.samples_size() > config_.max_fragment_samples) {
        *error = "fragment exceeds max_fragment_samples";
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
    const bool final_ends = final_sample.terminated() ||
                            final_sample.truncated();
    if (batch.trajectory_end() != final_ends) {
        *error = "trajectory_end does not match the final transition";
        return false;
    }
    if (final_sample.terminated()) {
        if (batch.bootstrap_valid() ||
            std::fabs(batch.bootstrap_value()) > 1e-6f) {
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

bool SamplePoolCoordinator::DeliveryBelongsToInstanceLocked(
    const std::string& delivery_id) const {
    return delivery_id.rfind(instance_id_ + "-delivery-", 0) == 0;
}

bool SamplePoolCoordinator::CapacityAllowsLocked(
    int64_t samples,
    int64_t fragments,
    int64_t estimated_bytes) const {
    return resident_samples_ + samples <= config_.max_queue_samples &&
           resident_fragments_ + fragments <= config_.max_queue_fragments &&
           resident_estimated_bytes_ + estimated_bytes <=
               config_.max_queue_estimated_bytes;
}

bool SamplePoolCoordinator::CanMakeCapacityLocked(
    int64_t samples,
    int64_t fragments,
    int64_t estimated_bytes) const {
    if (samples > config_.max_queue_samples ||
        fragments > config_.max_queue_fragments ||
        estimated_bytes > config_.max_queue_estimated_bytes) {
        return false;
    }
    const int64_t protected_samples = resident_samples_ - ready_samples_;
    const int64_t protected_fragments = resident_fragments_ - ready_fragments_;
    const int64_t protected_bytes =
        resident_estimated_bytes_ - ready_estimated_bytes_;
    return protected_samples + samples <= config_.max_queue_samples &&
           protected_fragments + fragments <= config_.max_queue_fragments &&
           protected_bytes + estimated_bytes <=
               config_.max_queue_estimated_bytes;
}

void SamplePoolCoordinator::EvictReadyUntilCapacityLocked(
    int64_t samples,
    int64_t fragments,
    int64_t estimated_bytes) {
    while (!CapacityAllowsLocked(samples, fragments, estimated_bytes)) {
        StoredFragment evicted = backend_->EvictOldestReady();
        ready_samples_ -= evicted.sample_count;
        --ready_fragments_;
        ready_estimated_bytes_ -= evicted.estimated_bytes;
        resident_samples_ -= evicted.sample_count;
        --resident_fragments_;
        resident_estimated_bytes_ -= evicted.estimated_bytes;
        evicted_sample_count_ += evicted.sample_count;
        ++evicted_fragment_count_;

        auto& counters = policy_counters_[evicted.policy_key];
        counters.ready_samples -= evicted.sample_count;
        --counters.ready_fragments;
        const auto active =
            active_batch_fingerprints_.find(evicted.batch.batch_id());
        if (active != active_batch_fingerprints_.end()) {
            RememberCompletedBatchLocked(evicted.batch.batch_id(),
                                         active->second);
            active_batch_fingerprints_.erase(active);
        }
    }
}

rl::training::v1::PressureState
SamplePoolCoordinator::PressureStateLocked() const {
    const double sample_ratio = static_cast<double>(resident_samples_) /
                                config_.max_queue_samples;
    const double fragment_ratio = static_cast<double>(resident_fragments_) /
                                  config_.max_queue_fragments;
    const double byte_ratio = static_cast<double>(resident_estimated_bytes_) /
                              config_.max_queue_estimated_bytes;
    const double ratio = std::max({sample_ratio, fragment_ratio, byte_ratio});
    if (ratio >= 1.0) return rl::training::v1::PRESSURE_STATE_FULL;
    if (ratio >= config_.high_watermark_ratio) {
        return rl::training::v1::PRESSURE_STATE_HIGH;
    }
    return rl::training::v1::PRESSURE_STATE_NORMAL;
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
    identity->mutable_source_digest()->set_algorithm(
        rl::common::v1::DIGEST_ALGORITHM_SHA256);
    identity->mutable_source_digest()->set_hex(config_.contract.source_digest);
    identity->mutable_artifact_digest()->set_algorithm(
        rl::common::v1::DIGEST_ALGORITHM_SHA256);
    identity->mutable_artifact_digest()->set_hex(
        config_.contract.artifact_digest);
    identity->set_platform(config_.contract.platform);
    identity->set_generator_identity(config_.contract.generator_identity);
}

void SamplePoolCoordinator::RememberDeliveryLocked(
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

void SamplePoolCoordinator::RememberCompletedBatchLocked(
    const std::string& batch_id,
    const SampleBatchFingerprint& fingerprint) {
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

const SamplePoolCoordinator::DeliveryRecord*
SamplePoolCoordinator::DeliveryHistoryLocked(
    const std::string& delivery_id) const {
    const auto found = delivery_history_.find(delivery_id);
    return found == delivery_history_.end() ? nullptr : &found->second;
}

bool SamplePoolCoordinator::PolicyMatchesFreshnessLocked(
    const rl::training::v1::BehaviorPolicyReference& policy,
    const rl::training::v1::SampleFreshnessPolicy& freshness) const {
    if (policy.model_lineage_id() != freshness.model_lineage_id() ||
        policy.distribution_schema_id() !=
            freshness.distribution_schema_id() ||
        policy.policy_spec_digest().SerializeAsString() !=
            freshness.policy_spec_digest().SerializeAsString() ||
        !policy.has_model_step() ||
        !freshness.has_reference_model_step() ||
        policy.model_step() > freshness.reference_model_step()) {
        return false;
    }
    return freshness.reference_model_step() - policy.model_step() <=
           freshness.max_model_step_lag();
}

bool SamplePoolCoordinator::FragmentEligibleLocked(
    const StoredFragment& fragment,
    const rl::training::v1::SampleFreshnessPolicy& freshness,
    const rl::training::v1::TrainingSemanticsIdentity& semantics,
    int64_t minimum_created_at_unix_ms) const {
    return fragment.batch.created_at_unix_ms() >=
               minimum_created_at_unix_ms &&
           fragment.batch.training_semantics().SerializeAsString() ==
               semantics.SerializeAsString() &&
           PolicyMatchesFreshnessLocked(fragment.batch.behavior_policy(),
                                        freshness);
}

std::vector<SamplePoolCoordinator::PolicyAvailability>
SamplePoolCoordinator::EligibleAvailabilityLocked(
    const rl::training::v1::SampleFreshnessPolicy& freshness,
    const rl::training::v1::TrainingSemanticsIdentity& semantics,
    int64_t minimum_created_at_unix_ms) const {
    std::map<std::string, PolicyAvailability> by_policy;
    size_t fifo_index = 0;
    for (const auto& fragment : backend_->ready()) {
        if (FragmentEligibleLocked(fragment, freshness, semantics,
                                   minimum_created_at_unix_ms)) {
            auto [iterator, inserted] = by_policy.emplace(
                fragment.policy_key,
                PolicyAvailability{fragment.policy_key, 0, fifo_index});
            iterator->second.sample_count += fragment.sample_count;
            if (inserted) iterator->second.first_fifo_index = fifo_index;
        }
        ++fifo_index;
    }
    std::vector<PolicyAvailability> availability;
    availability.reserve(by_policy.size());
    for (auto& [key, value] : by_policy) {
        (void)key;
        availability.push_back(std::move(value));
    }
    return availability;
}

std::string SamplePoolCoordinator::OldestEligiblePolicyLocked(
    const rl::training::v1::SampleFreshnessPolicy& freshness,
    const rl::training::v1::TrainingSemanticsIdentity& semantics,
    int64_t minimum_created_at_unix_ms,
    int64_t minimum_samples) const {
    std::string selected;
    size_t selected_index = std::numeric_limits<size_t>::max();
    for (const auto& available : EligibleAvailabilityLocked(
             freshness, semantics, minimum_created_at_unix_ms)) {
        if (available.sample_count < minimum_samples) continue;
        if (available.first_fifo_index < selected_index) {
            selected = available.policy_key;
            selected_index = available.first_fifo_index;
        }
    }
    return selected;
}

void SamplePoolCoordinator::Push(
    const rl::training::v1::PushSamplesReq& request,
    rl::training::v1::PushSamplesRsp* response) {
    response->Clear();
    const auto& batch = request.batch();
    const int64_t sample_count = CountSamples(batch);
    const int64_t estimated_bytes = EstimateBytes(batch);
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    ++push_attempt_count_;
    response->set_batch_id(batch.batch_id());
    FillServiceIdentity(response->mutable_sample_pool());

    std::string error;
    rl::training::v1::PushResult validation_rejection =
        rl::training::v1::PUSH_RESULT_REJECTED_INVALID;
    if (finalized_) {
        ++rejected_push_attempt_count_;
        rejected_sample_attempts_ += sample_count;
        last_error_ = "sample pool is finalized";
        response->set_ret_code(1);
        response->set_message(last_error_);
        response->set_result(
            rl::training::v1::PUSH_RESULT_REJECTED_FINALIZED);
    } else if (!ValidateBatchLocked(batch, &error, &validation_rejection)) {
        ++rejected_push_attempt_count_;
        rejected_sample_attempts_ += sample_count;
        last_error_ = error;
        response->set_ret_code(-1);
        response->set_message(error);
        response->set_result(validation_rejection);
    } else {
        const SampleBatchFingerprint fingerprint = FingerprintBatch(batch);
        const auto active = active_batch_fingerprints_.find(batch.batch_id());
        const auto completed =
            completed_batch_fingerprints_.find(batch.batch_id());
        if (active != active_batch_fingerprints_.end() ||
            completed != completed_batch_fingerprints_.end()) {
            const SampleBatchFingerprint& prior =
                active != active_batch_fingerprints_.end() ? active->second
                                                            : completed->second;
            if (prior == fingerprint) {
                ++duplicate_push_attempt_count_;
                duplicate_sample_attempts_ += sample_count;
                last_error_.clear();
                response->set_ret_code(0);
                response->set_message("batch already accepted");
                response->set_result(
                    rl::training::v1::PUSH_RESULT_DUPLICATE);
            } else {
                ++rejected_push_attempt_count_;
                rejected_sample_attempts_ += sample_count;
                last_error_ = "batch_id conflicts with a prior payload";
                response->set_ret_code(-1);
                response->set_message(last_error_);
                response->set_result(
                    rl::training::v1::PUSH_RESULT_REJECTED_CONFLICT);
            }
        } else {
            const std::string policy_key =
                PolicyKey(batch.behavior_policy(), batch.training_semantics());
            const auto known_policy = behavior_policy_by_key_.find(policy_key);
            const auto known_semantics =
                training_semantics_by_key_.find(policy_key);
            const bool policy_conflict =
                known_policy != behavior_policy_by_key_.end() &&
                (known_policy->second.SerializeAsString() !=
                     batch.behavior_policy().SerializeAsString() ||
                 known_semantics == training_semantics_by_key_.end() ||
                 known_semantics->second.SerializeAsString() !=
                     batch.training_semantics().SerializeAsString());
            if (policy_conflict) {
                ++rejected_push_attempt_count_;
                rejected_sample_attempts_ += sample_count;
                last_error_ = "policy key conflicts with behavior identity";
                response->set_ret_code(-1);
                response->set_message(last_error_);
                response->set_result(
                    rl::training::v1::PUSH_RESULT_REJECTED_IDENTITY);
            } else if (!CanMakeCapacityLocked(sample_count, 1,
                                               estimated_bytes)) {
                ++rejected_push_attempt_count_;
                rejected_sample_attempts_ += sample_count;
                last_error_ =
                    "fragment exceeds capacity protected by the active lease";
                response->set_ret_code(-1);
                response->set_message(last_error_);
                response->set_result(
                    rl::training::v1::PUSH_RESULT_REJECTED_CAPACITY);
            } else {
                EvictReadyUntilCapacityLocked(sample_count, 1,
                                               estimated_bytes);
                behavior_policy_by_key_[policy_key] = batch.behavior_policy();
                training_semantics_by_key_[policy_key] =
                    batch.training_semantics();
                StoredFragment stored;
                stored.batch = batch;
                stored.fingerprint = fingerprint;
                stored.policy_key = policy_key;
                stored.sample_count = sample_count;
                stored.estimated_bytes = estimated_bytes;
                backend_->PushBack(std::move(stored));

                auto& counters = policy_counters_[policy_key];
                counters.behavior_policy = batch.behavior_policy();
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
                last_error_.clear();
                response->set_ret_code(0);
                response->set_message("accepted");
                response->set_result(
                    rl::training::v1::PUSH_RESULT_ACCEPTED);
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

void SamplePoolCoordinator::GetBatch(
    const rl::training::v1::GetBatchReq& request,
    rl::training::v1::GetBatchRsp* response,
    const std::function<bool()>& is_cancelled) {
    response->Clear();
    const int target_samples = request.assembly().target_samples();
    const int max_samples = request.assembly().max_samples();
    const int timeout_ms = request.timeout_ms() > 0
                               ? request.timeout_ms()
                               : config_.default_get_timeout_ms;
    const int lease_timeout_ms = request.lease_timeout_ms() > 0
                                     ? request.lease_timeout_ms()
                                     : config_.default_lease_timeout_ms;
    const auto mode = request.assembly().mode();
    const auto start = std::chrono::steady_clock::now();
    const auto wait_deadline = start + std::chrono::milliseconds(timeout_ms);

    std::unique_lock<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    FillServiceIdentity(response->mutable_sample_pool());
    if (finalized_) {
        response->set_ret_code(-1);
        response->set_result(
            rl::training::v1::GET_BATCH_RESULT_REJECTED);
        response->set_message("sample pool is finalized");
        response->set_queue_size(ready_samples_);
        return;
    }
    std::string semantics_error;
    if (!IsServiceIdentityValid(request.consumer()) || target_samples <= 0 ||
        max_samples < target_samples || timeout_ms <= 0 ||
        lease_timeout_ms <= 0 ||
        request.freshness().model_lineage_id().empty() ||
        !request.freshness().has_reference_model_step() ||
        request.freshness().distribution_schema_id().empty() ||
        !IsSha256(request.freshness().policy_spec_digest()) ||
        request.freshness().max_sample_age_ms() <= 0 ||
        !ValidateSemantics(request.required_semantics(), &semantics_error) ||
        request.freshness().distribution_schema_id() !=
            request.required_semantics().policy_distribution_schema_id() ||
        (mode != rl::training::v1::BATCH_ASSEMBLY_MODE_TARGET_BOUNDED &&
         mode != rl::training::v1::BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE) ||
        (mode == rl::training::v1::BATCH_ASSEMBLY_MODE_TARGET_BOUNDED &&
         max_samples - target_samples + 1 < config_.max_fragment_samples)) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::GET_BATCH_RESULT_REJECTED);
        response->set_message("invalid GetBatch request");
        response->set_queue_size(ready_samples_);
        return;
    }

    const auto cancellation_requested = [&]() {
        return is_cancelled && is_cancelled();
    };
    const auto wait_deadline_reached = [&]() {
        return std::chrono::steady_clock::now() >= wait_deadline;
    };
    const auto minimum_created_at = [&]() {
        const int64_t now = NowMs();
        const int64_t maximum_age = request.freshness().max_sample_age_ms();
        return maximum_age >= now ? int64_t{0} : now - maximum_age;
    };
    const auto finish_without_lease = [&](bool cancelled) {
        response->Clear();
        FillServiceIdentity(response->mutable_sample_pool());
        if (cancelled) {
            response->set_ret_code(-1);
            response->set_result(
                rl::training::v1::GET_BATCH_RESULT_REJECTED);
            response->set_message("request cancelled before lease commit");
        } else {
            ++empty_timeout_count_;
            response->set_ret_code(1);
            response->set_result(
                rl::training::v1::GET_BATCH_RESULT_TIMEOUT);
            response->set_message(
                "compatible samples not available before deadline");
        }
        response->set_queue_size(ready_samples_);
        response->set_wait_ms(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start)
                .count());
    };
    const auto abort_before_lease = [&]() {
        const bool cancelled = cancellation_requested();
        if (!cancelled && !wait_deadline_reached()) return false;
        finish_without_lease(cancelled);
        return true;
    };

    if (abort_before_lease()) return;
    if (has_lease_) {
        ++consumer_busy_count_;
        response->set_ret_code(2);
        response->set_result(rl::training::v1::GET_BATCH_RESULT_BUSY);
        response->set_message("another delivery is still leased");
        response->set_queue_size(ready_samples_);
        response->set_leased_samples(lease_.sample_count);
        return;
    }

    const auto selected_policy = [&]() {
        return OldestEligiblePolicyLocked(
            request.freshness(), request.required_semantics(),
            minimum_created_at(),
            mode == rl::training::v1::BATCH_ASSEMBLY_MODE_TARGET_BOUNDED
                ? target_samples
                : 1);
    };

    std::string policy_key = selected_policy();
    while (policy_key.empty() &&
           std::chrono::steady_clock::now() < wait_deadline) {
        if (abort_before_lease()) return;
        cv_.wait_until(lock,
                       std::min(wait_deadline,
                                std::chrono::steady_clock::now() +
                                    std::chrono::milliseconds(50)));
        ReclaimExpiredLeaseLocked();
        if (finalized_) {
            response->Clear();
            FillServiceIdentity(response->mutable_sample_pool());
            response->set_ret_code(-1);
            response->set_result(
                rl::training::v1::GET_BATCH_RESULT_REJECTED);
            response->set_message("sample pool is finalized");
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
        policy_key = selected_policy();
    }
    if (abort_before_lease()) return;
    if (policy_key.empty()) {
        finish_without_lease(false);
        return;
    }

    const int64_t created_floor = minimum_created_at();
    std::deque<StoredFragment> preview;
    int64_t preview_samples = 0;
    int64_t preview_bytes = 0;
    for (const auto& fragment : backend_->ready()) {
        const bool still_collecting =
            mode == rl::training::v1::BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE ||
            preview_samples < target_samples;
        if (!still_collecting || fragment.policy_key != policy_key ||
            !FragmentEligibleLocked(fragment, request.freshness(),
                                    request.required_semantics(),
                                    created_floor) ||
            preview_samples + fragment.sample_count > max_samples) {
            continue;
        }
        preview_samples += fragment.sample_count;
        preview_bytes += fragment.estimated_bytes;
        preview.push_back(fragment);
    }
    if (preview.empty() ||
        (mode == rl::training::v1::BATCH_ASSEMBLY_MODE_TARGET_BOUNDED &&
         preview_samples < target_samples)) {
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::GET_BATCH_RESULT_REJECTED);
        response->set_message("bounded assembly invariant violated");
        response->set_queue_size(ready_samples_);
        return;
    }

    uint64_t minimum_step = 0;
    uint64_t maximum_step = 0;
    int64_t oldest_created_at = 0;
    int64_t newest_created_at = 0;
    bool first_fragment = true;
    for (const auto& fragment : preview) {
        *response->add_batches() = fragment.batch;
        const uint64_t step = fragment.batch.behavior_policy().model_step();
        const int64_t created_at = fragment.batch.created_at_unix_ms();
        if (first_fragment) {
            minimum_step = maximum_step = step;
            oldest_created_at = newest_created_at = created_at;
            first_fragment = false;
        } else {
            minimum_step = std::min(minimum_step, step);
            maximum_step = std::max(maximum_step, step);
            oldest_created_at = std::min(oldest_created_at, created_at);
            newest_created_at = std::max(newest_created_at, created_at);
        }
    }

    const bool cancelled_before_commit = cancellation_requested();
    if (cancelled_before_commit || wait_deadline_reached()) {
        finish_without_lease(cancelled_before_commit);
        return;
    }

    std::deque<StoredFragment> selected = backend_->ExtractPolicy(
        policy_key, created_floor, target_samples, max_samples,
        mode == rl::training::v1::BATCH_ASSEMBLY_MODE_DRAIN_AVAILABLE);
    if (selected.size() != preview.size()) {
        backend_->RestoreFront(std::move(selected));
        response->Clear();
        FillServiceIdentity(response->mutable_sample_pool());
        response->set_ret_code(-1);
        response->set_result(rl::training::v1::GET_BATCH_RESULT_REJECTED);
        response->set_message("storage selection changed before lease commit");
        response->set_queue_size(ready_samples_);
        return;
    }

    Lease new_lease;
    new_lease.consumer_instance_id = ServiceKey(request.consumer());
    new_lease.fragments = std::move(selected);
    new_lease.sample_count = preview_samples;
    new_lease.estimated_bytes = preview_bytes;
    new_lease.delivery_id = instance_id_ + "-delivery-" +
                            std::to_string(next_delivery_sequence_++);
    new_lease.deadline = std::chrono::steady_clock::now() +
                         std::chrono::milliseconds(lease_timeout_ms);
    new_lease.deadline_unix_ms = NowMs() + lease_timeout_ms;

    for (const auto& fragment : new_lease.fragments) {
        auto& counters = policy_counters_[fragment.policy_key];
        counters.ready_samples -= fragment.sample_count;
        --counters.ready_fragments;
        counters.leased_samples += fragment.sample_count;
        ++counters.leased_fragments;
    }
    ready_samples_ -= new_lease.sample_count;
    ready_fragments_ -= static_cast<int64_t>(new_lease.fragments.size());
    ready_estimated_bytes_ -= new_lease.estimated_bytes;

    if (new_lease.sample_count >= target_samples) {
        ++target_hit_count_;
    } else {
        ++partial_get_count_;
    }
    response->set_ret_code(0);
    response->set_result(rl::training::v1::GET_BATCH_RESULT_LEASED);
    response->set_message("leased");
    response->set_delivery_id(new_lease.delivery_id);
    response->set_lease_deadline_unix_ms(new_lease.deadline_unix_ms);
    response->set_returned_samples(new_lease.sample_count);
    response->set_actual_batch_size(new_lease.sample_count);
    response->set_returned_fragments(
        static_cast<int64_t>(new_lease.fragments.size()));
    response->set_queue_size(ready_samples_);
    response->set_leased_samples(new_lease.sample_count);
    response->set_wait_ms(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    response->set_minimum_behavior_model_step(minimum_step);
    response->set_maximum_behavior_model_step(maximum_step);
    response->set_oldest_sample_created_at_unix_ms(oldest_created_at);
    response->set_newest_sample_created_at_unix_ms(newest_created_at);
    lease_ = std::move(new_lease);
    has_lease_ = true;
}

void SamplePoolCoordinator::FillDeliveryResponseLocked(
    rl::training::v1::DeliveryRsp* response) const {
    response->set_queue_size(ready_samples_);
    if (has_lease_) {
        response->set_lease_deadline_unix_ms(lease_.deadline_unix_ms);
    }
}

void SamplePoolCoordinator::FillFinalizeResponseLocked(
    rl::training::v1::FinalizeSamplePoolRsp* response) const {
    FillServiceIdentity(response->mutable_sample_pool());
    response->set_settled_samples(finalized_sample_count_);
    response->set_settled_fragments(finalized_fragment_count_);
    response->set_ready_samples(ready_samples_);
    response->set_ready_fragments(ready_fragments_);
    response->set_leased_samples(has_lease_ ? lease_.sample_count : 0);
    response->set_leased_fragments(
        has_lease_ ? static_cast<int64_t>(lease_.fragments.size()) : 0);
    response->set_resident_samples(resident_samples_);
    response->set_resident_fragments(resident_fragments_);
    response->set_finalized_at_unix_ms(finalized_at_unix_ms_);
}

void SamplePoolCoordinator::RequeueLeaseLocked(bool expired) {
    if (!has_lease_) return;
    for (const auto& fragment : lease_.fragments) {
        auto& counters = policy_counters_[fragment.policy_key];
        counters.ready_samples += fragment.sample_count;
        ++counters.ready_fragments;
        counters.leased_samples -= fragment.sample_count;
        --counters.leased_fragments;
    }
    ready_samples_ += lease_.sample_count;
    ready_fragments_ += static_cast<int64_t>(lease_.fragments.size());
    ready_estimated_bytes_ += lease_.estimated_bytes;
    backend_->RestoreFront(std::move(lease_.fragments));

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

void SamplePoolCoordinator::ReclaimExpiredLeaseLocked() {
    if (has_lease_ && std::chrono::steady_clock::now() >= lease_.deadline) {
        RequeueLeaseLocked(true);
    }
}

void SamplePoolCoordinator::Ack(
    const rl::training::v1::AckBatchReq& request,
    rl::training::v1::DeliveryRsp* response) {
    response->Clear();
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
    const int64_t fragments =
        static_cast<int64_t>(lease_.fragments.size());
    for (const auto& fragment : lease_.fragments) {
        auto& counters = policy_counters_[fragment.policy_key];
        counters.leased_samples -= fragment.sample_count;
        --counters.leased_fragments;
        counters.acked_samples += fragment.sample_count;
        ++counters.acked_fragments;
        switch (request.disposition()) {
            case rl::training::v1::ACK_DISPOSITION_TRAINED:
                counters.trained_samples += fragment.sample_count;
                break;
            case rl::training::v1::ACK_DISPOSITION_STALE:
                counters.stale_samples += fragment.sample_count;
                break;
            case rl::training::v1::ACK_DISPOSITION_INVALID:
                counters.invalid_samples += fragment.sample_count;
                break;
            case rl::training::v1::ACK_DISPOSITION_SHUTDOWN_UNTRAINED:
                counters.shutdown_untrained_samples += fragment.sample_count;
                break;
            default:
                break;
        }
    }
    switch (request.disposition()) {
        case rl::training::v1::ACK_DISPOSITION_TRAINED:
            trained_sample_count_ += samples;
            break;
        case rl::training::v1::ACK_DISPOSITION_STALE:
            stale_sample_count_ += samples;
            break;
        case rl::training::v1::ACK_DISPOSITION_INVALID:
            invalid_sample_count_ += samples;
            break;
        case rl::training::v1::ACK_DISPOSITION_SHUTDOWN_UNTRAINED:
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
    for (const auto& fragment : lease_.fragments) {
        const auto active =
            active_batch_fingerprints_.find(fragment.batch.batch_id());
        if (active != active_batch_fingerprints_.end()) {
            RememberCompletedBatchLocked(fragment.batch.batch_id(),
                                         active->second);
            active_batch_fingerprints_.erase(active);
        }
    }
    latest_ack_unix_ms_ = NowMs();
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

void SamplePoolCoordinator::Nack(
    const rl::training::v1::NackBatchReq& request,
    rl::training::v1::DeliveryRsp* response) {
    response->Clear();
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
            history->result == rl::training::v1::DELIVERY_RESULT_EXPIRED ? 1
                                                                         : 0);
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

void SamplePoolCoordinator::RenewLease(
    const rl::training::v1::RenewLeaseReq& request,
    rl::training::v1::DeliveryRsp* response) {
    response->Clear();
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
        response->set_result(
            history->result == rl::training::v1::DELIVERY_RESULT_EXPIRED
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

void SamplePoolCoordinator::FinalizeSamplePool(
    const rl::training::v1::FinalizeSamplePoolReq& request,
    rl::training::v1::FinalizeSamplePoolRsp* response) {
    response->Clear();
    std::lock_guard<std::mutex> lock(mutex_);
    ReclaimExpiredLeaseLocked();
    response->set_finalization_id(request.finalization_id());
    FillFinalizeResponseLocked(response);

    const auto& expected = request.expected_sample_pool();
    const bool identity_valid =
        IsServiceIdentityValid(request.consumer()) &&
        request.consumer().component() == "learner" &&
        IsServiceIdentityValid(expected) &&
        expected.component() == "sample-pool" &&
        expected.instance_id() == instance_id_ &&
        expected.lifecycle_epoch() == 1 &&
        !request.finalization_id().empty() &&
        request.finalization_id().size() <= 256;
    if (!identity_valid) {
        response->set_ret_code(-1);
        response->set_result(
            rl::training::v1::
                SAMPLE_POOL_FINALIZE_RESULT_REJECTED_IDENTITY);
        response->set_message("invalid SamplePool finalization identity");
        return;
    }

    const std::string consumer_key = ServiceKey(request.consumer());
    if (finalized_) {
        response->set_finalization_id(finalization_id_);
        FillFinalizeResponseLocked(response);
        if (request.finalization_id() == finalization_id_ &&
            consumer_key == finalization_consumer_key_) {
            response->set_ret_code(0);
            response->set_result(
                rl::training::v1::
                    SAMPLE_POOL_FINALIZE_RESULT_ALREADY_FINALIZED);
            response->set_message("sample pool already finalized");
        } else {
            response->set_ret_code(-1);
            response->set_result(
                rl::training::v1::
                    SAMPLE_POOL_FINALIZE_RESULT_REJECTED_CONFLICT);
            response->set_message(
                "SamplePool finalization conflicts with applied identity");
        }
        return;
    }

    if (has_lease_) {
        response->set_ret_code(1);
        response->set_result(
            rl::training::v1::
                SAMPLE_POOL_FINALIZE_RESULT_REJECTED_ACTIVE_LEASE);
        response->set_message(
            "active delivery must settle before SamplePool finalization");
        return;
    }

    std::deque<StoredFragment> ready = backend_->ExtractAllReady();
    int64_t extracted_samples = 0;
    int64_t extracted_bytes = 0;
    for (const auto& fragment : ready) {
        extracted_samples += fragment.sample_count;
        extracted_bytes += fragment.estimated_bytes;
    }
    const int64_t extracted_fragments =
        static_cast<int64_t>(ready.size());
    if (extracted_samples != ready_samples_ ||
        extracted_fragments != ready_fragments_ ||
        extracted_bytes != ready_estimated_bytes_ ||
        resident_samples_ != ready_samples_ ||
        resident_fragments_ != ready_fragments_ ||
        resident_estimated_bytes_ != ready_estimated_bytes_) {
        backend_->RestoreFront(std::move(ready));
        last_error_ = "SamplePool ready-tail accounting is inconsistent";
        response->set_ret_code(-1);
        response->set_result(
            rl::training::v1::
                SAMPLE_POOL_FINALIZE_RESULT_REJECTED_CONFLICT);
        response->set_message(last_error_);
        return;
    }

    for (const auto& fragment : ready) {
        auto& counters = policy_counters_[fragment.policy_key];
        counters.ready_samples -= fragment.sample_count;
        --counters.ready_fragments;
        counters.acked_samples += fragment.sample_count;
        ++counters.acked_fragments;
        counters.shutdown_untrained_samples += fragment.sample_count;

        const auto active =
            active_batch_fingerprints_.find(fragment.batch.batch_id());
        if (active != active_batch_fingerprints_.end()) {
            RememberCompletedBatchLocked(fragment.batch.batch_id(),
                                         active->second);
            active_batch_fingerprints_.erase(active);
        }
    }

    ready_samples_ = 0;
    ready_fragments_ = 0;
    ready_estimated_bytes_ = 0;
    resident_samples_ = 0;
    resident_fragments_ = 0;
    resident_estimated_bytes_ = 0;
    acked_unique_samples_ += extracted_samples;
    acked_unique_batches_ += extracted_fragments;
    shutdown_untrained_sample_count_ += extracted_samples;
    latest_ack_unix_ms_ = NowMs();
    finalized_ = true;
    finalization_id_ = request.finalization_id();
    finalization_consumer_key_ = consumer_key;
    finalized_at_unix_ms_ = latest_ack_unix_ms_;
    finalized_sample_count_ = extracted_samples;
    finalized_fragment_count_ = extracted_fragments;
    last_error_.clear();

    response->Clear();
    response->set_ret_code(0);
    response->set_result(
        rl::training::v1::SAMPLE_POOL_FINALIZE_RESULT_FINALIZED);
    response->set_message("sample pool finalized");
    response->set_finalization_id(finalization_id_);
    FillFinalizeResponseLocked(response);
    cv_.notify_all();
}

void SamplePoolCoordinator::FillStatusScalarsLocked(
    rl::training::v1::SamplePoolStatusRsp* response) const {
    FillContractIdentity(response->mutable_contract());
    FillServiceIdentity(response->mutable_sample_pool());
    response->set_ready(!finalized_);
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
        has_lease_ ? static_cast<int64_t>(lease_.fragments.size()) : 0);
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
    response->set_ingress_ready(!finalized_);
    response->set_pool_ready(!finalized_);
    response->set_evicted_sample_count(evicted_sample_count_);
    response->set_evicted_fragment_count(evicted_fragment_count_);
    response->set_finalized(finalized_);
    response->set_finalization_id(finalization_id_);
    response->set_finalized_at_unix_ms(finalized_at_unix_ms_);
    response->set_finalized_sample_count(finalized_sample_count_);
    response->set_finalized_fragment_count(finalized_fragment_count_);

    const int64_t snapshot_time = NowMs();
    response->set_timestamp_unix_ms(snapshot_time);
    bool has_ready = false;
    int64_t oldest_created_at = 0;
    uint64_t minimum_step = 0;
    uint64_t maximum_step = 0;
    for (const auto& fragment : backend_->ready()) {
        const int64_t created_at = fragment.batch.created_at_unix_ms();
        const uint64_t step = fragment.batch.behavior_policy().model_step();
        if (!has_ready) {
            oldest_created_at = created_at;
            minimum_step = maximum_step = step;
            has_ready = true;
        } else {
            oldest_created_at = std::min(oldest_created_at, created_at);
            minimum_step = std::min(minimum_step, step);
            maximum_step = std::max(maximum_step, step);
        }
    }
    if (has_ready) {
        response->set_oldest_ready_sample_age_ms(
            std::max<int64_t>(0, snapshot_time - oldest_created_at));
        response->set_minimum_ready_model_step(minimum_step);
        response->set_maximum_ready_model_step(maximum_step);
    }
}

void SamplePoolCoordinator::AppendBehaviorSteps(
    const std::vector<PolicyCounters>& policy_snapshot,
    rl::training::v1::SamplePoolStatusRsp* response) {
    for (const auto& counters : policy_snapshot) {
        auto* status = response->add_behavior_steps();
        *status->mutable_behavior_policy() = counters.behavior_policy;
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
}

void SamplePoolCoordinator::GetStatus(
    const rl::training::v1::SamplePoolStatusReq&,
    rl::training::v1::SamplePoolStatusRsp* response) {
    response->Clear();
    std::vector<PolicyCounters> policy_snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ReclaimExpiredLeaseLocked();
        policy_snapshot.reserve(policy_counters_.size());
        for (const auto& [key, counters] : policy_counters_) {
            (void)key;
            policy_snapshot.push_back(counters);
        }
        FillStatusScalarsLocked(response);
    }
    AppendBehaviorSteps(policy_snapshot, response);
}

const std::string& SamplePoolCoordinator::instance_id() const {
    return instance_id_;
}
