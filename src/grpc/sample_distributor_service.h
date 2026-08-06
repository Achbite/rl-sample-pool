#pragma once

#include "config/config_loader.h"
#include "training.grpc.pb.h"
#include "store/sample_store.h"

class SampleDistributorServiceImpl final : public rl::training::v1::SampleDistributorService::Service {
public:
    explicit SampleDistributorServiceImpl(const DistributorConfig& config);

    grpc::Status PushSamples(grpc::ServerContext* ctx,
                             const rl::training::v1::SampleBatch* req,
                             rl::training::v1::PushSamplesRsp* rsp) override;

    grpc::Status GetBatch(grpc::ServerContext* ctx,
                          const rl::training::v1::GetBatchReq* req,
                          rl::training::v1::GetBatchRsp* rsp) override;

    grpc::Status AckBatch(grpc::ServerContext* ctx,
                          const rl::training::v1::AckBatchReq* req,
                          rl::training::v1::DeliveryRsp* rsp) override;

    grpc::Status NackBatch(grpc::ServerContext* ctx,
                           const rl::training::v1::NackBatchReq* req,
                           rl::training::v1::DeliveryRsp* rsp) override;

    grpc::Status RenewLease(grpc::ServerContext* ctx,
                            const rl::training::v1::RenewLeaseReq* req,
                            rl::training::v1::DeliveryRsp* rsp) override;

    grpc::Status GetStatus(grpc::ServerContext* ctx,
                           const rl::training::v1::DistributorStatusReq* req,
                           rl::training::v1::DistributorStatusRsp* rsp) override;

    const std::string& instance_id() const;

private:
    SampleStore store_;
};
