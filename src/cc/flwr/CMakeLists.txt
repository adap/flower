# Copyright
#
# cmake build file for C++ flower client example.
# Assumes protobuf and gRPC have been installed using cmake.
# that automatically builds all the dependencies before building flwr client

cmake_minimum_required(VERSION 3.5.1)

project(Flower C CXX)
set (CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "/home/jiang/libtorch;/home/jiang/torchvision_cpp;/home/jiang/grpc_lib")

find_package(Threads REQUIRED)

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
set(_REFLECTION gRPC::grpc++_reflection)


if(CMAKE_CROSSCOMPILING)
  find_program(_PROTOBUF_PROTOC protoc)
else()
  set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
endif()

find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(_GRPC_GRPCPP gRPC::grpc++)
if(CMAKE_CROSSCOMPILING)
  find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)
endif()

find_package(TorchVision REQUIRED)

# Proto file
get_filename_component(trans_proto "../../proto/flwr/proto/transport.proto" ABSOLUTE)
get_filename_component(trans_proto_path "${trans_proto}" PATH)

# Generated sources
set(trans_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/transport.pb.cc")
set(trans_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/transport.pb.h")
set(trans_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/transport.grpc.pb.cc")
set(trans_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/transport.grpc.pb.h")
add_custom_command(
          OUTPUT "${trans_proto_srcs}" "${trans_proto_hdrs}" "${trans_grpc_srcs}" "${trans_grpc_hdrs}"
          COMMAND ${_PROTOBUF_PROTOC}
          ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
          --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
          -I "${trans_proto_path}"
          --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
          "${trans_proto}"
          DEPENDS "${trans_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# trans_grpc_proto
add_library(trans_grpc_proto
  ${trans_grpc_srcs}
  ${trans_grpc_hdrs}
  ${trans_proto_srcs}
  ${trans_proto_hdrs}
  "message_handler.h"
  "serde.h"
  "client.h"
  "pytorch_client.h"
  "start.h"
  "typing.h")
target_link_libraries(trans_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})


find_package(TorchVision REQUIRED)

add_library(helper
	"cifar10.h"
        "cifar10.cpp"
	"transform.h"
	"transform.cpp")
target_link_libraries(helper
	TorchVision::TorchVision)

# Targets flower client
set(_target flwr_client)
add_executable(${_target} "${_target}.cc")

#target_include_directories(${EXECUTABLE_NAME} ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${_target}
     trans_grpc_proto
     helper)

if(DOWNLOAD_DATASETS)
  add_dependencies(${_target} cifar10)
endif()

