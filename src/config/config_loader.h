#pragma once

#include <cstdint>
#include <string>

struct DistributorConfig {
    int listen_port = 9100;
    std::string run_id = "local-run";
    int64_t max_queue_samples = 262144;
    int64_t max_queue_fragments = 4096;
    int64_t max_queue_estimated_bytes = 536870912;
    int64_t max_dedup_entries = 1000000;
    double high_watermark_ratio = 0.8;
    int default_get_timeout_ms = 1000;
    int default_lease_timeout_ms = 10000;
    int delivery_history_size = 4096;
};

bool LoadDistributorConfig(const std::string& yaml_path, DistributorConfig& out_config);
