#include "config/config_loader.h"
#include "grpc/sample_pool_service.h"

#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>

static const char* kDefaultConfigPath = "configs/pool_config.yaml";

int main(int argc, char* argv[]) {
    std::cout << "============================================\n";
    std::cout << "  RL Training - Local Sample Pool\n";
    std::cout << "============================================\n\n";

    const char* config_path = kDefaultConfigPath;
    if (argc > 1) {
        config_path = argv[1];
    }

    SamplePoolConfig config;
    if (!LoadSamplePoolConfig(config_path, config)) {
        return 2;
    }

    SamplePoolCoordinator coordinator(config);
    SamplePoolIngressServiceImpl ingress_service(coordinator);
    SamplePoolConsumerServiceImpl consumer_service(coordinator);
    std::string listen_addr = "0.0.0.0:" + std::to_string(config.listen_port);

    grpc::ServerBuilder builder;
    builder.AddListeningPort(listen_addr, grpc::InsecureServerCredentials());
    builder.RegisterService(&ingress_service);
    builder.RegisterService(&consumer_service);

    std::unique_ptr<grpc::Server> server = builder.BuildAndStart();
    if (!server) {
        std::cerr << "[SamplePool] failed to start: " << listen_addr
                  << std::endl;
        return 1;
    }

    std::cout << "[SamplePool] listening on " << listen_addr << std::endl;
    std::cout << "[SamplePool] backend=" << config.backend_type
              << ", max_concurrent_consumers=1" << std::endl;
    std::cout << "[SamplePool] instance_id="
              << coordinator.instance_id() << std::endl;
    std::cout << "[SamplePool] capacity_transitions="
              << config.capacity_transitions
              << ", capacity_bytes="
              << config.capacity_bytes
              << ", sampling=uniform_without_replacement"
              << ", sampling_seed=" << config.sampling_seed
              << ", eviction=fifo_ready"
              << ", default_get_timeout_ms="
              << config.default_get_timeout_ms
              << ", default_lease_timeout_ms="
              << config.default_lease_timeout_ms << std::endl;

    server->Wait();
    return 0;
}
