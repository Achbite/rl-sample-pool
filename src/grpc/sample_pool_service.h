#pragma once

#include "store/sample_pool_coordinator.h"
#include "training.grpc.pb.h"

class SamplePoolIngressServiceImpl final
    : public rl::training::v1::SamplePoolIngressService::Service {
public:
    explicit SamplePoolIngressServiceImpl(SamplePoolCoordinator& coordinator);

    grpc::Status PushSamples(
        grpc::ServerContext* context,
        const rl::training::v1::PushSamplesReq* request,
        rl::training::v1::PushSamplesRsp* response) override;
    grpc::Status GetStatus(
        grpc::ServerContext* context,
        const rl::training::v1::SamplePoolStatusReq* request,
        rl::training::v1::SamplePoolStatusRsp* response) override;

private:
    SamplePoolCoordinator& coordinator_;
};

class SamplePoolConsumerServiceImpl final
    : public rl::training::v1::SamplePoolConsumerService::Service {
public:
    explicit SamplePoolConsumerServiceImpl(SamplePoolCoordinator& coordinator);

    grpc::Status GetBatch(
        grpc::ServerContext* context,
        const rl::training::v1::GetBatchReq* request,
        rl::training::v1::GetBatchRsp* response) override;
    grpc::Status AckBatch(
        grpc::ServerContext* context,
        const rl::training::v1::AckBatchReq* request,
        rl::training::v1::DeliveryRsp* response) override;
    grpc::Status NackBatch(
        grpc::ServerContext* context,
        const rl::training::v1::NackBatchReq* request,
        rl::training::v1::DeliveryRsp* response) override;
    grpc::Status RenewLease(
        grpc::ServerContext* context,
        const rl::training::v1::RenewLeaseReq* request,
        rl::training::v1::DeliveryRsp* response) override;
    grpc::Status FinalizeSamplePool(
        grpc::ServerContext* context,
        const rl::training::v1::FinalizeSamplePoolReq* request,
        rl::training::v1::FinalizeSamplePoolRsp* response) override;
    grpc::Status GetStatus(
        grpc::ServerContext* context,
        const rl::training::v1::SamplePoolStatusReq* request,
        rl::training::v1::SamplePoolStatusRsp* response) override;

private:
    SamplePoolCoordinator& coordinator_;
};
