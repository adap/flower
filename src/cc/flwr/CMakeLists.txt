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

cmake_minimum_required(VERSION 3.5.1)

project(Flower C CXX)

include(../cmake/common.cmake)

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
  ${trans_proto_hdrs})
target_link_libraries(trans_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

# Targets flower client
foreach(_target
  flwr_client)
  add_executable(${_target} "${_target}.cc")
  target_sources(${_target} PRIVATE cifar10.cpp
	                            transform.cpp
	                            cifar10.h
                               	    transform.h
  )
  target_link_libraries(${_target}
     trans_grpc_proto
     ${_REFLECTION}
     ${_GRPC_GRPCPP}
     ${_PROTOBUF_LIBPROTOBUF}
     TorchVision::TorchVision)
  if(DOWNLOAD_DATASETS)
    add_dependencies(${_target} cifar10)
  endif()
endforeach()


