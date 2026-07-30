#pragma once

#include "config/config_loader.h"
#include "maze.grpc.pb.h"
#include "store/sample_store.h"

class SampleDistributorServiceImpl final : public maze::SampleDistributorService::Service {
public:
    explicit SampleDistributorServiceImpl(const DistributorConfig& config);

    grpc::Status PushSamples(grpc::ServerContext* ctx,
                             const maze::SampleBatch* req,
                             maze::PushSamplesRsp* rsp) override;

    grpc::Status GetBatch(grpc::ServerContext* ctx,
                          const maze::GetBatchReq* req,
                          maze::GetBatchRsp* rsp) override;

    grpc::Status AckBatch(grpc::ServerContext* ctx,
                          const maze::AckBatchReq* req,
                          maze::DeliveryRsp* rsp) override;

    grpc::Status NackBatch(grpc::ServerContext* ctx,
                           const maze::NackBatchReq* req,
                           maze::DeliveryRsp* rsp) override;

    grpc::Status RenewLease(grpc::ServerContext* ctx,
                            const maze::RenewLeaseReq* req,
                            maze::DeliveryRsp* rsp) override;

    grpc::Status GetStatus(grpc::ServerContext* ctx,
                           const maze::DistributorStatusReq* req,
                           maze::DistributorStatusRsp* rsp) override;

    const std::string& instance_id() const;

private:
    SampleStore store_;
};
