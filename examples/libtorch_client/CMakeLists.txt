# Copyright 2018 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# cmake build file for C++ helloworld example.
# Assumes protobuf and gRPC have been installed using cmake.
# See cmake_externalproject/CMakeLists.txt for all-in-one cmake build
# that automatically builds all the dependencies before building helloworld.

cmake_minimum_required(VERSION 3.15)
project(LibTorchClient VERSION 0.10
        DESCRIPTION "Creates a Libtorch client that trains a ResNet18 models on CIFAR10."
        LANGUAGES CXX)

# Choose existing gRPC or build a new one
include(cmake/common.cmake)

# Include Torchvision, which includes Torch 
find_package(TorchVision REQUIRED)

# Protobuf file defining flower messages
get_filename_component(trans_proto "../../src/proto/flwr/proto/transport.proto" ABSOLUTE)
get_filename_component(trans_proto_path "${trans_proto}" PATH)

## gRPC 
# Generated C++ sources and headers from protobuff
set(trans_proto_srcs "transport.pb.cc")
set(trans_proto_hdrs "transport.pb.h")
set(trans_grpc_srcs "transport.grpc.pb.cc")
set(trans_grpc_hdrs "transport.grpc.pb.h")

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
include_directories("../../src/cc/flwr/include")
include_directories(include)

# Build trans_grpc_proto
add_library(trans_grpc_proto
  STATIC
  ${trans_grpc_srcs}
  ${trans_grpc_hdrs}
  ${trans_proto_srcs}
  ${trans_proto_hdrs})

target_include_directories(trans_grpc_proto PUBLIC include)

target_link_libraries(trans_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# Targets flower client
add_executable(client "src/main.cc"
                      "src/cifar10.cc"
                      "src/torch_client.cc"
                      "src/transform.cc"
                      "include/torch_client.h"
                      "include/cifar10.h"
                      "../../src/cc/flwr/include/client.h"
                      "../../src/cc/flwr/include/start.h"
                      "include/transform.h")
target_link_libraries(client
     trans_grpc_proto
     gRPC::grpc++_reflection
     ${_GRPC_GRPCPP}
     ${_PROTOBUF_LIBPROTOBUF}
     TorchVision::TorchVision)

if(DOWNLOAD_DATASETS)
   add_dependencies(client cifar10)
 endif()
#add_executable(flwr_client src/client.cc)
#target_include_directories(flwr_client include)
#target_link_libraries(flwr_client grpc++)