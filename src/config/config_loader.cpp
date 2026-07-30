#include "config/config_loader.h"

#include <fstream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static std::string Trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::string StripQuotes(const std::string& s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') ||
         (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

static int SafeInt(const std::string& val, int def) {
    if (val.empty()) return def;
    try {
        return std::stoi(val);
    } catch (...) {
        return def;
    }
}

static int64_t SafeInt64(const std::string& val, int64_t def) {
    if (val.empty()) return def;
    try {
        return std::stoll(val);
    } catch (...) {
        return def;
    }
}

static double SafeDouble(const std::string& val, double def) {
    if (val.empty()) return def;
    try {
        return std::stod(val);
    } catch (...) {
        return def;
    }
}

static std::string GetEnvValue(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : "";
}

struct YamlEntry {
    std::string section;
    std::string key;
    std::string value;
};

static std::vector<YamlEntry> ParseYaml(const std::string& text) {
    std::vector<YamlEntry> entries;
    std::istringstream stream(text);
    std::string line;
    std::string current_section;

    while (std::getline(stream, line)) {
        size_t comment_pos = std::string::npos;
        bool in_quotes = false;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '"' || line[i] == '\'') {
                in_quotes = !in_quotes;
            } else if (line[i] == '#' && !in_quotes) {
                comment_pos = i;
                break;
            }
        }
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        std::string trimmed = Trim(line);
        if (trimmed.empty()) continue;

        size_t colon_pos = trimmed.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key_part = Trim(trimmed.substr(0, colon_pos));
        std::string val_part = Trim(trimmed.substr(colon_pos + 1));
        bool has_indent = (!line.empty() && (line[0] == ' ' || line[0] == '\t'));

        if (!has_indent && val_part.empty()) {
            current_section = key_part;
        } else if (has_indent && !key_part.empty()) {
            entries.push_back(YamlEntry{current_section, key_part, StripQuotes(val_part)});
        }
    }

    return entries;
}

static std::string FindValue(const std::vector<YamlEntry>& entries,
                             const std::string& section,
                             const std::string& key) {
    for (const auto& entry : entries) {
        if (entry.section == section && entry.key == key) {
            return entry.value;
        }
    }
    return "";
}

bool LoadDistributorConfig(const std::string& yaml_path, DistributorConfig& out_config) {
    std::ifstream ifs(yaml_path);
    if (!ifs.is_open()) {
        std::cerr << "[SampleDistributor] config not found, using defaults: "
                  << yaml_path << std::endl;
        out_config = DistributorConfig{};
        return false;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    std::vector<YamlEntry> entries = ParseYaml(ss.str());

    out_config.listen_port = SafeInt(FindValue(entries, "server", "listen_port"), 9100);
    std::string run_id = FindValue(entries, "server", "run_id");
    if (!run_id.empty()) {
        out_config.run_id = run_id;
    }

    out_config.max_queue_samples = SafeInt64(
        FindValue(entries, "queue", "max_queue_samples"), 262144);
    out_config.max_queue_fragments = SafeInt64(
        FindValue(entries, "queue", "max_queue_fragments"), 4096);
    out_config.max_queue_estimated_bytes = SafeInt64(
        FindValue(entries, "queue", "max_queue_estimated_bytes"), 536870912);
    out_config.max_dedup_entries = SafeInt64(
        FindValue(entries, "queue", "max_dedup_entries"), 1000000);
    out_config.high_watermark_ratio = SafeDouble(
        FindValue(entries, "queue", "high_watermark_ratio"), 0.8);
    out_config.default_get_timeout_ms = SafeInt(
        FindValue(entries, "queue", "default_get_timeout_ms"), 1000);
    out_config.default_lease_timeout_ms = SafeInt(
        FindValue(entries, "queue", "default_lease_timeout_ms"), 10000);
    out_config.delivery_history_size = SafeInt(
        FindValue(entries, "queue", "delivery_history_size"), 4096);

    std::string env_port = GetEnvValue("MAZE_SAMPLE_DISTRIBUTOR_PORT");
    if (!env_port.empty()) {
        out_config.listen_port = SafeInt(env_port, out_config.listen_port);
    }
    std::string env_run_id = GetEnvValue("MAZE_RUN_ID");
    if (!env_run_id.empty()) {
        out_config.run_id = env_run_id;
    }
    std::string env_max_samples = GetEnvValue("MAZE_DISTRIBUTOR_MAX_SAMPLES");
    if (!env_max_samples.empty()) {
        out_config.max_queue_samples =
            SafeInt64(env_max_samples, out_config.max_queue_samples);
    }
    std::string env_max_fragments = GetEnvValue("MAZE_DISTRIBUTOR_MAX_FRAGMENTS");
    if (!env_max_fragments.empty()) {
        out_config.max_queue_fragments =
            SafeInt64(env_max_fragments, out_config.max_queue_fragments);
    }
    std::string env_max_bytes = GetEnvValue("MAZE_DISTRIBUTOR_MAX_BYTES");
    if (!env_max_bytes.empty()) {
        out_config.max_queue_estimated_bytes =
            SafeInt64(env_max_bytes, out_config.max_queue_estimated_bytes);
    }
    std::string env_lease_timeout = GetEnvValue("MAZE_DISTRIBUTOR_LEASE_TIMEOUT_MS");
    if (!env_lease_timeout.empty()) {
        out_config.default_lease_timeout_ms =
            SafeInt(env_lease_timeout, out_config.default_lease_timeout_ms);
    }

    if (out_config.max_queue_samples <= 0) {
        out_config.max_queue_samples = 262144;
    }
    if (out_config.max_queue_fragments <= 0) {
        out_config.max_queue_fragments = 4096;
    }
    if (out_config.max_queue_estimated_bytes <= 0) {
        out_config.max_queue_estimated_bytes = 536870912;
    }
    if (out_config.max_dedup_entries <= 0) {
        out_config.max_dedup_entries = 1000000;
    }
    if (out_config.high_watermark_ratio <= 0.0 ||
        out_config.high_watermark_ratio >= 1.0) {
        out_config.high_watermark_ratio = 0.8;
    }
    if (out_config.default_get_timeout_ms <= 0) {
        out_config.default_get_timeout_ms = 1000;
    }
    if (out_config.default_lease_timeout_ms <= 0) {
        out_config.default_lease_timeout_ms = 10000;
    }
    if (out_config.delivery_history_size <= 0) {
        out_config.delivery_history_size = 4096;
    }
    return true;
}
