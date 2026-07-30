#include "grpc/sample_distributor_service.h"

SampleDistributorServiceImpl::SampleDistributorServiceImpl(
    const DistributorConfig& config)
    : store_(config) {}

grpc::Status SampleDistributorServiceImpl::PushSamples(
    grpc::ServerContext*,
    const maze::SampleBatch* request,
    maze::PushSamplesRsp* response) {
    store_.Push(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::GetBatch(
    grpc::ServerContext* context,
    const maze::GetBatchReq* request,
    maze::GetBatchRsp* response) {
    store_.GetBatch(
        *request, response, [context]() { return context->IsCancelled(); });
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::AckBatch(
    grpc::ServerContext*,
    const maze::AckBatchReq* request,
    maze::DeliveryRsp* response) {
    store_.Ack(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::NackBatch(
    grpc::ServerContext*,
    const maze::NackBatchReq* request,
    maze::DeliveryRsp* response) {
    store_.Nack(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::RenewLease(
    grpc::ServerContext*,
    const maze::RenewLeaseReq* request,
    maze::DeliveryRsp* response) {
    store_.RenewLease(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::GetStatus(
    grpc::ServerContext*,
    const maze::DistributorStatusReq* request,
    maze::DistributorStatusRsp* response) {
    store_.GetStatus(*request, response);
    return grpc::Status::OK;
}

const std::string& SampleDistributorServiceImpl::instance_id() const {
    return store_.instance_id();
}
