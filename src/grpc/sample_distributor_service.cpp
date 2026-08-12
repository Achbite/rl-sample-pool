#include "grpc/sample_distributor_service.h"

#include <chrono>

SampleDistributorServiceImpl::SampleDistributorServiceImpl(
    const DistributorConfig& config)
    : store_(config) {}

grpc::Status SampleDistributorServiceImpl::UpsertSampleDemand(
    grpc::ServerContext*,
    const rl::training::v1::UpsertSampleDemandReq* request,
    rl::training::v1::SampleDemandRsp* response) {
    store_.UpsertDemand(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::ReleaseSampleDemand(
    grpc::ServerContext*,
    const rl::training::v1::ReleaseSampleDemandReq* request,
    rl::training::v1::SampleDemandRsp* response) {
    store_.ReleaseDemand(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::GetSampleDemandStatus(
    grpc::ServerContext*,
    const rl::training::v1::GetSampleDemandStatusReq* request,
    rl::training::v1::SampleDemandStatusRsp* response) {
    store_.GetDemandStatus(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::AcquireSampleCredit(
    grpc::ServerContext*,
    const rl::training::v1::AcquireSampleCreditReq* request,
    rl::training::v1::SampleCreditGrant* response) {
    store_.AcquireCredit(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::ReleaseSampleCredit(
    grpc::ServerContext*,
    const rl::training::v1::ReleaseSampleCreditReq* request,
    rl::training::v1::ReleaseSampleCreditRsp* response) {
    store_.ReleaseCredit(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::PushSamples(
    grpc::ServerContext*,
    const rl::training::v1::PushSamplesReq* request,
    rl::training::v1::PushSamplesRsp* response) {
    store_.Push(*request, response);
    return grpc::Status::OK;
}

grpc::Status SampleDistributorServiceImpl::GetBatch(
    grpc::ServerContext* context,
    const rl::training::v1::GetBatchReq* request,
    rl::training::v1::GetBatchRsp* response) {
    const auto rpc_deadline = context->deadline();
    store_.GetBatch(
        *request, response, [context, rpc_deadline]() {
            return context->IsCancelled() ||
                   std::chrono::system_clock::now() >= rpc_deadline;
        });
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
