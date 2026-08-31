#include "config/config_loader.h"

#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string Trim(const std::string& value) {
    const size_t start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

std::string StripQuotes(const std::string& value) {
    if (value.size() >= 2 &&
        ((value.front() == '"' && value.back() == '"') ||
         (value.front() == '\'' && value.back() == '\''))) {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

struct YamlEntry {
    std::string section;
    std::string key;
    std::string value;
};

bool ParseYaml(const std::string& text,
               std::vector<YamlEntry>* entries,
               std::string* error) {
    std::istringstream stream(text);
    std::string line;
    std::string section;
    int line_number = 0;
    while (std::getline(stream, line)) {
        ++line_number;
        bool in_quotes = false;
        for (size_t index = 0; index < line.size(); ++index) {
            if (line[index] == '"' || line[index] == '\'') {
                in_quotes = !in_quotes;
            } else if (line[index] == '#' && !in_quotes) {
                line.erase(index);
                break;
            }
        }
        const std::string trimmed = Trim(line);
        if (trimmed.empty()) continue;
        const size_t colon = trimmed.find(':');
        if (colon == std::string::npos) {
            *error = "line " + std::to_string(line_number) +
                     " is not a key/value entry";
            return false;
        }
        const std::string key = Trim(trimmed.substr(0, colon));
        const std::string value = Trim(trimmed.substr(colon + 1));
        const bool indented = !line.empty() &&
                              (line.front() == ' ' || line.front() == '\t');
        if (!indented && value.empty()) {
            section = key;
            continue;
        }
        if (!indented || section.empty() || key.empty() || value.empty()) {
            *error = "line " + std::to_string(line_number) +
                     " has unsupported YAML structure";
            return false;
        }
        entries->push_back({section, key, StripQuotes(value)});
    }
    return true;
}

const std::string* Find(const std::vector<YamlEntry>& entries,
                        const std::string& section,
                        const std::string& key) {
    const std::string* found = nullptr;
    for (const auto& entry : entries) {
        if (entry.section == section && entry.key == key) {
            if (found != nullptr) return nullptr;
            found = &entry.value;
        }
    }
    return found;
}

template <typename T>
bool ParseInteger(const std::string& value, T* output) {
    try {
        size_t consumed = 0;
        const long long parsed = std::stoll(value, &consumed);
        if (consumed != value.size()) return false;
        *output = static_cast<T>(parsed);
        return static_cast<long long>(*output) == parsed;
    } catch (...) {
        return false;
    }
}

bool ParseDouble(const std::string& value, double* output) {
    try {
        size_t consumed = 0;
        *output = std::stod(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

bool IsLowerSha256(const std::string& value) {
    if (value.size() != 64) return false;
    for (char character : value) {
        if (!((character >= '0' && character <= '9') ||
              (character >= 'a' && character <= 'f'))) {
            return false;
        }
    }
    return true;
}

bool IsAllowedEntry(const YamlEntry& entry) {
    static const std::set<std::string> allowed = {
        "server.listen_port",
        "backend.type",
        "storage.capacity_transitions",
        "storage.capacity_bytes",
        "sampling.seed",
        "sampling.strategy",
        "eviction.strategy",
        "queue.max_dedup_entries",
        "queue.high_watermark_ratio",
        "queue.default_get_timeout_ms",
        "queue.default_lease_timeout_ms",
        "queue.delivery_history_size",
        "contract.package_name",
        "contract.package_version",
        "contract.source_digest",
        "contract.artifact_digest",
        "contract.platform",
        "contract.generator_identity",
    };
    return allowed.count(entry.section + "." + entry.key) == 1;
}

}  // namespace

bool LoadSamplePoolConfig(const std::string& yaml_path,
                          SamplePoolConfig& output) {
    std::ifstream input(yaml_path);
    if (!input.is_open()) {
        std::cerr << "[SamplePool] required config not found: " << yaml_path
                  << std::endl;
        return false;
    }
    std::stringstream buffer;
    buffer << input.rdbuf();
    std::vector<YamlEntry> entries;
    std::string error;
    if (!ParseYaml(buffer.str(), &entries, &error)) {
        std::cerr << "[SamplePool] invalid config: " << error << std::endl;
        return false;
    }
    for (const auto& entry : entries) {
        if (!IsAllowedEntry(entry)) {
            std::cerr << "[SamplePool] unknown config key: " << entry.section
                      << "." << entry.key << std::endl;
            return false;
        }
    }

    const auto required = [&](const char* section,
                              const char* key) -> const std::string* {
        const std::string* value = Find(entries, section, key);
        if (value == nullptr || value->empty()) {
            std::cerr << "[SamplePool] missing or duplicate config key: "
                      << section << "." << key << std::endl;
            return nullptr;
        }
        return value;
    };

    SamplePoolConfig parsed;
    const std::string* listen_port = required("server", "listen_port");
    const std::string* backend_type = required("backend", "type");
    const std::string* capacity_transitions =
        required("storage", "capacity_transitions");
    const std::string* capacity_bytes = required("storage", "capacity_bytes");
    const std::string* sampling_seed = required("sampling", "seed");
    const std::string* sampling_strategy =
        required("sampling", "strategy");
    const std::string* eviction_strategy =
        required("eviction", "strategy");
    const std::string* max_dedup = required("queue", "max_dedup_entries");
    const std::string* high_watermark =
        required("queue", "high_watermark_ratio");
    const std::string* get_timeout =
        required("queue", "default_get_timeout_ms");
    const std::string* lease_timeout =
        required("queue", "default_lease_timeout_ms");
    const std::string* delivery_history =
        required("queue", "delivery_history_size");
    const std::string* package_name = required("contract", "package_name");
    const std::string* package_version =
        required("contract", "package_version");
    const std::string* source_digest = required("contract", "source_digest");
    const std::string* artifact_digest =
        required("contract", "artifact_digest");
    const std::string* platform = required("contract", "platform");
    const std::string* generator =
        required("contract", "generator_identity");
    if (!listen_port || !backend_type || !capacity_transitions ||
        !capacity_bytes || !sampling_seed || !sampling_strategy ||
        !eviction_strategy ||
        !max_dedup || !high_watermark || !get_timeout || !lease_timeout ||
        !delivery_history || !package_name || !package_version ||
        !source_digest || !artifact_digest || !platform || !generator) {
        return false;
    }
    if (!ParseInteger(*listen_port, &parsed.listen_port) ||
        !ParseInteger(*capacity_transitions, &parsed.capacity_transitions) ||
        !ParseInteger(*capacity_bytes, &parsed.capacity_bytes) ||
        !ParseInteger(*sampling_seed, &parsed.sampling_seed) ||
        !ParseInteger(*max_dedup, &parsed.max_dedup_entries) ||
        !ParseDouble(*high_watermark, &parsed.high_watermark_ratio) ||
        !ParseInteger(*get_timeout, &parsed.default_get_timeout_ms) ||
        !ParseInteger(*lease_timeout, &parsed.default_lease_timeout_ms) ||
        !ParseInteger(*delivery_history, &parsed.delivery_history_size)) {
        std::cerr << "[SamplePool] a numeric config value is malformed"
                  << std::endl;
        return false;
    }
    parsed.backend_type = *backend_type;
    parsed.contract.package_name = *package_name;
    parsed.contract.package_version = *package_version;
    parsed.contract.source_digest = *source_digest;
    parsed.contract.artifact_digest = *artifact_digest;
    parsed.contract.platform = *platform;
    parsed.contract.generator_identity = *generator;
    if (parsed.listen_port <= 0 || parsed.listen_port > 65535 ||
        parsed.backend_type != "local_memory" ||
        parsed.capacity_transitions <= 0 || parsed.capacity_bytes <= 0 ||
        *sampling_strategy != "uniform_without_replacement" ||
        *eviction_strategy != "fifo_ready" ||
        parsed.max_dedup_entries <= 0 ||
        parsed.high_watermark_ratio <= 0.0 ||
        parsed.high_watermark_ratio >= 1.0 ||
        !std::isfinite(parsed.high_watermark_ratio) ||
        parsed.default_get_timeout_ms <= 0 ||
        parsed.default_lease_timeout_ms <= 0 ||
        parsed.delivery_history_size <= 0 ||
        parsed.contract.package_name != "rl-contracts" ||
        parsed.contract.package_version != "0.15.0" ||
        !IsLowerSha256(parsed.contract.source_digest) ||
        !IsLowerSha256(parsed.contract.artifact_digest) ||
        parsed.contract.platform.empty() ||
        !IsLowerSha256(parsed.contract.generator_identity)) {
        std::cerr << "[SamplePool] config value is outside the locked contract"
                  << std::endl;
        return false;
    }

    // Deployment wiring and bounded local-store controls may be overridden by
    // the component launcher. Backend and sampling semantics remain explicit
    // config facts; unsupported values fail instead of selecting a fallback.
    if (const char* port = std::getenv("RL_SAMPLE_POOL_PORT")) {
        int override_port = 0;
        if (*port == '\0' ||
            !ParseInteger(std::string(port), &override_port) ||
            override_port <= 0 || override_port > 65535) {
            std::cerr << "[SamplePool] RL_SAMPLE_POOL_PORT is invalid"
                      << std::endl;
            return false;
        }
        parsed.listen_port = override_port;
    }
    if (const char* capacity =
            std::getenv("RL_SAMPLE_POOL_CAPACITY_TRANSITIONS")) {
        if (*capacity == '\0' ||
            !ParseInteger(std::string(capacity),
                          &parsed.capacity_transitions) ||
            parsed.capacity_transitions <= 0) {
            std::cerr
                << "[SamplePool] RL_SAMPLE_POOL_CAPACITY_TRANSITIONS is invalid"
                << std::endl;
            return false;
        }
    }
    if (const char* capacity = std::getenv("RL_SAMPLE_POOL_CAPACITY_BYTES")) {
        if (*capacity == '\0' ||
            !ParseInteger(std::string(capacity), &parsed.capacity_bytes) ||
            parsed.capacity_bytes <= 0) {
            std::cerr << "[SamplePool] RL_SAMPLE_POOL_CAPACITY_BYTES is invalid"
                      << std::endl;
            return false;
        }
    }
    if (const char* seed = std::getenv("RL_SAMPLE_POOL_SAMPLING_SEED")) {
        if (*seed == '\0' ||
            !ParseInteger(std::string(seed), &parsed.sampling_seed)) {
            std::cerr << "[SamplePool] RL_SAMPLE_POOL_SAMPLING_SEED is invalid"
                      << std::endl;
            return false;
        }
    }
    output = parsed;
    return true;
}
