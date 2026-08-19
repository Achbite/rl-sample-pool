#pragma once

#include <cstdint>
#include <string>

struct ContractConfig {
    std::string package_name;
    std::string package_version;
    std::string source_digest;
    std::string artifact_digest;
    std::string platform;
    std::string generator_identity;
};

struct SamplePoolConfig {
    int listen_port = 9100;
    int64_t max_queue_samples = 262144;
    int64_t max_queue_fragments = 4096;
    int max_fragment_samples = 128;
    int64_t max_queue_estimated_bytes = 536870912;
    int64_t max_dedup_entries = 1000000;
    double high_watermark_ratio = 0.8;
    int default_get_timeout_ms = 1000;
    int default_lease_timeout_ms = 10000;
    int delivery_history_size = 4096;
    ContractConfig contract;
};

bool LoadSamplePoolConfig(const std::string& yaml_path,
                          SamplePoolConfig& out_config);
