#pragma once

#include <cstdint>
#include <string>

struct ContractConfig {
    std::string package_name;
    std::string package_version;
    std::string platform;
};

struct SamplePoolConfig {
    int listen_port = 9100;
    std::string backend_type = "local_memory";
    int64_t capacity_transitions = 10240;
    int64_t capacity_bytes = 536870912;
    uint64_t sampling_seed = 0;
    int64_t max_dedup_entries = 1000000;
    double high_watermark_ratio = 0.8;
    int default_get_timeout_ms = 1000;
    int default_lease_timeout_ms = 10000;
    int delivery_history_size = 4096;
    ContractConfig contract;
};

bool LoadSamplePoolConfig(const std::string& yaml_path,
                          SamplePoolConfig& out_config);
