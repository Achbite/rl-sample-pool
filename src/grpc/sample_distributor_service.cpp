#include "grpc/sample_distributor_service.h"

SampleDistributorServiceImpl::SampleDistributorServiceImpl(
    const DistributorConfig& config)
    : store_(config) {}

grpc::Status SampleDistributorServiceImpl::PushSamples(
    grpc::ServerContext*,
    const rl::training::v1::SampleBatch* request,
    rl::training::v1::PushSamplesRsp* response) {
    store_.Push(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::GetBatch(
    grpc::ServerContext* context,
    const rl::training::v1::GetBatchReq* request,
    rl::training::v1::GetBatchRsp* response) {
    store_.GetBatch(
        *request, response, [context]() { return context->IsCancelled(); });
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::AckBatch(
    grpc::ServerContext*,
    const rl::training::v1::AckBatchReq* request,
    rl::training::v1::DeliveryRsp* response) {
    store_.Ack(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::NackBatch(
    grpc::ServerContext*,
    const rl::training::v1::NackBatchReq* request,
    rl::training::v1::DeliveryRsp* response) {
    store_.Nack(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::RenewLease(
    grpc::ServerContext*,
    const rl::training::v1::RenewLeaseReq* request,
    rl::training::v1::DeliveryRsp* response) {
    store_.RenewLease(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::GetStatus(
    grpc::ServerContext*,
    const rl::training::v1::DistributorStatusReq* request,
    rl::training::v1::DistributorStatusRsp* response) {
    store_.GetStatus(*request, response);
    return grpc::Status::OK;
}

const std::string& SampleDistributorServiceImpl::instance_id() const {
    return store_.instance_id();
}
