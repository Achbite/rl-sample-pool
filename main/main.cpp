#include "config/config_loader.h"
#include "grpc/sample_distributor_service.h"

#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

static const char* kDefaultConfigPath = "configs/distributor_config.yaml";

int main(int argc, char* argv[]) {
    std::cout << "============================================\n";
    std::cout << "  Maze RL - SampleDistributor\n";
    std::cout << "============================================\n\n";

    const char* config_path = kDefaultConfigPath;
    if (argc > 1) {
        config_path = argv[1];
    }

    DistributorConfig config;
    LoadDistributorConfig(config_path, config);

    SampleDistributorServiceImpl service(config);
    std::string listen_addr = "0.0.0.0:" + std::to_string(config.listen_port);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server) {
        std::cerr << "[SampleDistributor] failed to start: " << listen_addr << std::endl;
        return 1;
    }

    std::cout << "[SampleDistributor] listening on " << listen_addr << std::endl;
    std::cout << "[SampleDistributor] instance_id="
              << service.instance_id() << std::endl;
    std::cout << "[SampleDistributor] max_queue_samples="
              << config.max_queue_samples
              << ", max_queue_fragments="
              << config.max_queue_fragments
              << ", max_queue_estimated_bytes="
              << config.max_queue_estimated_bytes
              << ", default_get_timeout_ms="
              << config.default_get_timeout_ms
              << ", default_lease_timeout_ms="
              << config.default_lease_timeout_ms << std::endl;

    server->Wait();
    return 0;
}
