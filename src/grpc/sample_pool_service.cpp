#include "grpc/sample_pool_service.h"

#include <chrono>

SamplePoolIngressServiceImpl::SamplePoolIngressServiceImpl(
    SamplePoolCoordinator& coordinator)
    : coordinator_(coordinator) {}

grpc::Status SamplePoolIngressServiceImpl::PushSamples(
    grpc::ServerContext*,
    const rl::training::v1::PushSamplesReq* request,
    rl::training::v1::PushSamplesRsp* response) {
    coordinator_.Push(*request, response);
    return grpc::Status::OK;
}

grpc::Status SamplePoolIngressServiceImpl::GetStatus(
    grpc::ServerContext*,
    const rl::training::v1::SamplePoolStatusReq* request,
    rl::training::v1::SamplePoolStatusRsp* response) {
    coordinator_.GetStatus(*request, response);
    return grpc::Status::OK;
}

SamplePoolConsumerServiceImpl::SamplePoolConsumerServiceImpl(
    SamplePoolCoordinator& coordinator)
    : coordinator_(coordinator) {}

grpc::Status SamplePoolConsumerServiceImpl::GetBatch(
    grpc::ServerContext* context,
    const rl::training::v1::GetBatchReq* request,
    rl::training::v1::GetBatchRsp* response) {
    const auto rpc_deadline = context->deadline();
    coordinator_.GetBatch(
        *request, response, [context, rpc_deadline]() {
            return context->IsCancelled() ||
                   std::chrono::system_clock::now() >= rpc_deadline;
        });
    return grpc::Status::OK;
}

grpc::Status SamplePoolConsumerServiceImpl::AckBatch(
    grpc::ServerContext*,
    const rl::training::v1::AckBatchReq* request,
    rl::training::v1::DeliveryRsp* response) {
    coordinator_.Ack(*request, response);
    return grpc::Status::OK;
}

grpc::Status SamplePoolConsumerServiceImpl::NackBatch(
    grpc::ServerContext*,
    const rl::training::v1::NackBatchReq* request,
    rl::training::v1::DeliveryRsp* response) {
    coordinator_.Nack(*request, response);
    return grpc::Status::OK;
}

grpc::Status SamplePoolConsumerServiceImpl::RenewLease(
    grpc::ServerContext*,
    const rl::training::v1::RenewLeaseReq* request,
    rl::training::v1::DeliveryRsp* response) {
    coordinator_.RenewLease(*request, response);
    return grpc::Status::OK;
}

grpc::Status SamplePoolConsumerServiceImpl::FinalizeSamplePool(
    grpc::ServerContext*,
    const rl::training::v1::FinalizeSamplePoolReq* request,
    rl::training::v1::FinalizeSamplePoolRsp* response) {
    coordinator_.FinalizeSamplePool(*request, response);
    return grpc::Status::OK;
}

grpc::Status SamplePoolConsumerServiceImpl::GetStatus(
    grpc::ServerContext*,
    const rl::training::v1::SamplePoolStatusReq* request,
    rl::training::v1::SamplePoolStatusRsp* response) {
    coordinator_.GetStatus(*request, response);
    return grpc::Status::OK;
}
