// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: flwr/proto/fleet.proto

#include "flwr/proto/fleet.pb.h"
#include "flwr/proto/fleet.grpc.pb.h"

#include <functional>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/channel_interface.h>
#include <grpcpp/impl/codegen/client_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/message_allocator.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/rpc_service_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/server_callback_handlers.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/sync_stream.h>
namespace flwr {
namespace proto {

static const char* Fleet_method_names[] = {
  "/flwr.proto.Fleet/CreateNode",
  "/flwr.proto.Fleet/DeleteNode",
  "/flwr.proto.Fleet/PullTaskIns",
  "/flwr.proto.Fleet/PushTaskRes",
};

std::unique_ptr< Fleet::Stub> Fleet::NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options) {
  (void)options;
  std::unique_ptr< Fleet::Stub> stub(new Fleet::Stub(channel, options));
  return stub;
}

Fleet::Stub::Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options)
  : channel_(channel), rpcmethod_CreateNode_(Fleet_method_names[0], options.suffix_for_stats(),::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_DeleteNode_(Fleet_method_names[1], options.suffix_for_stats(),::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_PullTaskIns_(Fleet_method_names[2], options.suffix_for_stats(),::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  , rpcmethod_PushTaskRes_(Fleet_method_names[3], options.suffix_for_stats(),::grpc::internal::RpcMethod::NORMAL_RPC, channel)
  {}

::grpc::Status Fleet::Stub::CreateNode(::grpc::ClientContext* context, const ::flwr::proto::CreateNodeRequest& request, ::flwr::proto::CreateNodeResponse* response) {
  return ::grpc::internal::BlockingUnaryCall< ::flwr::proto::CreateNodeRequest, ::flwr::proto::CreateNodeResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), rpcmethod_CreateNode_, context, request, response);
}

void Fleet::Stub::async::CreateNode(::grpc::ClientContext* context, const ::flwr::proto::CreateNodeRequest* request, ::flwr::proto::CreateNodeResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc::internal::CallbackUnaryCall< ::flwr::proto::CreateNodeRequest, ::flwr::proto::CreateNodeResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_CreateNode_, context, request, response, std::move(f));
}

void Fleet::Stub::async::CreateNode(::grpc::ClientContext* context, const ::flwr::proto::CreateNodeRequest* request, ::flwr::proto::CreateNodeResponse* response, ::grpc::ClientUnaryReactor* reactor) {
  ::grpc::internal::ClientCallbackUnaryFactory::Create< ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_CreateNode_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::flwr::proto::CreateNodeResponse>* Fleet::Stub::PrepareAsyncCreateNodeRaw(::grpc::ClientContext* context, const ::flwr::proto::CreateNodeRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderHelper::Create< ::flwr::proto::CreateNodeResponse, ::flwr::proto::CreateNodeRequest, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), cq, rpcmethod_CreateNode_, context, request);
}

::grpc::ClientAsyncResponseReader< ::flwr::proto::CreateNodeResponse>* Fleet::Stub::AsyncCreateNodeRaw(::grpc::ClientContext* context, const ::flwr::proto::CreateNodeRequest& request, ::grpc::CompletionQueue* cq) {
  auto* result =
    this->PrepareAsyncCreateNodeRaw(context, request, cq);
  result->StartCall();
  return result;
}

::grpc::Status Fleet::Stub::DeleteNode(::grpc::ClientContext* context, const ::flwr::proto::DeleteNodeRequest& request, ::flwr::proto::DeleteNodeResponse* response) {
  return ::grpc::internal::BlockingUnaryCall< ::flwr::proto::DeleteNodeRequest, ::flwr::proto::DeleteNodeResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), rpcmethod_DeleteNode_, context, request, response);
}

void Fleet::Stub::async::DeleteNode(::grpc::ClientContext* context, const ::flwr::proto::DeleteNodeRequest* request, ::flwr::proto::DeleteNodeResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc::internal::CallbackUnaryCall< ::flwr::proto::DeleteNodeRequest, ::flwr::proto::DeleteNodeResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_DeleteNode_, context, request, response, std::move(f));
}

void Fleet::Stub::async::DeleteNode(::grpc::ClientContext* context, const ::flwr::proto::DeleteNodeRequest* request, ::flwr::proto::DeleteNodeResponse* response, ::grpc::ClientUnaryReactor* reactor) {
  ::grpc::internal::ClientCallbackUnaryFactory::Create< ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_DeleteNode_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::flwr::proto::DeleteNodeResponse>* Fleet::Stub::PrepareAsyncDeleteNodeRaw(::grpc::ClientContext* context, const ::flwr::proto::DeleteNodeRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderHelper::Create< ::flwr::proto::DeleteNodeResponse, ::flwr::proto::DeleteNodeRequest, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), cq, rpcmethod_DeleteNode_, context, request);
}

::grpc::ClientAsyncResponseReader< ::flwr::proto::DeleteNodeResponse>* Fleet::Stub::AsyncDeleteNodeRaw(::grpc::ClientContext* context, const ::flwr::proto::DeleteNodeRequest& request, ::grpc::CompletionQueue* cq) {
  auto* result =
    this->PrepareAsyncDeleteNodeRaw(context, request, cq);
  result->StartCall();
  return result;
}

::grpc::Status Fleet::Stub::PullTaskIns(::grpc::ClientContext* context, const ::flwr::proto::PullTaskInsRequest& request, ::flwr::proto::PullTaskInsResponse* response) {
  return ::grpc::internal::BlockingUnaryCall< ::flwr::proto::PullTaskInsRequest, ::flwr::proto::PullTaskInsResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), rpcmethod_PullTaskIns_, context, request, response);
}

void Fleet::Stub::async::PullTaskIns(::grpc::ClientContext* context, const ::flwr::proto::PullTaskInsRequest* request, ::flwr::proto::PullTaskInsResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc::internal::CallbackUnaryCall< ::flwr::proto::PullTaskInsRequest, ::flwr::proto::PullTaskInsResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_PullTaskIns_, context, request, response, std::move(f));
}

void Fleet::Stub::async::PullTaskIns(::grpc::ClientContext* context, const ::flwr::proto::PullTaskInsRequest* request, ::flwr::proto::PullTaskInsResponse* response, ::grpc::ClientUnaryReactor* reactor) {
  ::grpc::internal::ClientCallbackUnaryFactory::Create< ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_PullTaskIns_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::flwr::proto::PullTaskInsResponse>* Fleet::Stub::PrepareAsyncPullTaskInsRaw(::grpc::ClientContext* context, const ::flwr::proto::PullTaskInsRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderHelper::Create< ::flwr::proto::PullTaskInsResponse, ::flwr::proto::PullTaskInsRequest, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), cq, rpcmethod_PullTaskIns_, context, request);
}

::grpc::ClientAsyncResponseReader< ::flwr::proto::PullTaskInsResponse>* Fleet::Stub::AsyncPullTaskInsRaw(::grpc::ClientContext* context, const ::flwr::proto::PullTaskInsRequest& request, ::grpc::CompletionQueue* cq) {
  auto* result =
    this->PrepareAsyncPullTaskInsRaw(context, request, cq);
  result->StartCall();
  return result;
}

::grpc::Status Fleet::Stub::PushTaskRes(::grpc::ClientContext* context, const ::flwr::proto::PushTaskResRequest& request, ::flwr::proto::PushTaskResResponse* response) {
  return ::grpc::internal::BlockingUnaryCall< ::flwr::proto::PushTaskResRequest, ::flwr::proto::PushTaskResResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), rpcmethod_PushTaskRes_, context, request, response);
}

void Fleet::Stub::async::PushTaskRes(::grpc::ClientContext* context, const ::flwr::proto::PushTaskResRequest* request, ::flwr::proto::PushTaskResResponse* response, std::function<void(::grpc::Status)> f) {
  ::grpc::internal::CallbackUnaryCall< ::flwr::proto::PushTaskResRequest, ::flwr::proto::PushTaskResResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_PushTaskRes_, context, request, response, std::move(f));
}

void Fleet::Stub::async::PushTaskRes(::grpc::ClientContext* context, const ::flwr::proto::PushTaskResRequest* request, ::flwr::proto::PushTaskResResponse* response, ::grpc::ClientUnaryReactor* reactor) {
  ::grpc::internal::ClientCallbackUnaryFactory::Create< ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(stub_->channel_.get(), stub_->rpcmethod_PushTaskRes_, context, request, response, reactor);
}

::grpc::ClientAsyncResponseReader< ::flwr::proto::PushTaskResResponse>* Fleet::Stub::PrepareAsyncPushTaskResRaw(::grpc::ClientContext* context, const ::flwr::proto::PushTaskResRequest& request, ::grpc::CompletionQueue* cq) {
  return ::grpc::internal::ClientAsyncResponseReaderHelper::Create< ::flwr::proto::PushTaskResResponse, ::flwr::proto::PushTaskResRequest, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(channel_.get(), cq, rpcmethod_PushTaskRes_, context, request);
}

::grpc::ClientAsyncResponseReader< ::flwr::proto::PushTaskResResponse>* Fleet::Stub::AsyncPushTaskResRaw(::grpc::ClientContext* context, const ::flwr::proto::PushTaskResRequest& request, ::grpc::CompletionQueue* cq) {
  auto* result =
    this->PrepareAsyncPushTaskResRaw(context, request, cq);
  result->StartCall();
  return result;
}

Fleet::Service::Service() {
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      Fleet_method_names[0],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< Fleet::Service, ::flwr::proto::CreateNodeRequest, ::flwr::proto::CreateNodeResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(
          [](Fleet::Service* service,
             ::grpc::ServerContext* ctx,
             const ::flwr::proto::CreateNodeRequest* req,
             ::flwr::proto::CreateNodeResponse* resp) {
               return service->CreateNode(ctx, req, resp);
             }, this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      Fleet_method_names[1],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< Fleet::Service, ::flwr::proto::DeleteNodeRequest, ::flwr::proto::DeleteNodeResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(
          [](Fleet::Service* service,
             ::grpc::ServerContext* ctx,
             const ::flwr::proto::DeleteNodeRequest* req,
             ::flwr::proto::DeleteNodeResponse* resp) {
               return service->DeleteNode(ctx, req, resp);
             }, this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      Fleet_method_names[2],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< Fleet::Service, ::flwr::proto::PullTaskInsRequest, ::flwr::proto::PullTaskInsResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(
          [](Fleet::Service* service,
             ::grpc::ServerContext* ctx,
             const ::flwr::proto::PullTaskInsRequest* req,
             ::flwr::proto::PullTaskInsResponse* resp) {
               return service->PullTaskIns(ctx, req, resp);
             }, this)));
  AddMethod(new ::grpc::internal::RpcServiceMethod(
      Fleet_method_names[3],
      ::grpc::internal::RpcMethod::NORMAL_RPC,
      new ::grpc::internal::RpcMethodHandler< Fleet::Service, ::flwr::proto::PushTaskResRequest, ::flwr::proto::PushTaskResResponse, ::grpc::protobuf::MessageLite, ::grpc::protobuf::MessageLite>(
          [](Fleet::Service* service,
             ::grpc::ServerContext* ctx,
             const ::flwr::proto::PushTaskResRequest* req,
             ::flwr::proto::PushTaskResResponse* resp) {
               return service->PushTaskRes(ctx, req, resp);
             }, this)));
}

Fleet::Service::~Service() {
}

::grpc::Status Fleet::Service::CreateNode(::grpc::ServerContext* context, const ::flwr::proto::CreateNodeRequest* request, ::flwr::proto::CreateNodeResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status Fleet::Service::DeleteNode(::grpc::ServerContext* context, const ::flwr::proto::DeleteNodeRequest* request, ::flwr::proto::DeleteNodeResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status Fleet::Service::PullTaskIns(::grpc::ServerContext* context, const ::flwr::proto::PullTaskInsRequest* request, ::flwr::proto::PullTaskInsResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}

::grpc::Status Fleet::Service::PushTaskRes(::grpc::ServerContext* context, const ::flwr::proto::PushTaskResRequest* request, ::flwr::proto::PushTaskResResponse* response) {
  (void) context;
  (void) request;
  (void) response;
  return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
}


}  // namespace flwr
}  // namespace proto

