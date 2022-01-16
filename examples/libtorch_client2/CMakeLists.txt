cmake_minimum_required(VERSION 3.15)
project(LIBTORCHCLIENT VERSION 0.10
  DESCRIPTION "Creates a LibTorch client that trains a ResNet18 model on CIFAR10."
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Add dependencies
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
include(FetchContent)
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.43.0
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(gRPC)

include(FindProtobuf)
find_package(Protobuf CONFIG REQUIRED)

set(EXECUTABLE_NAME client)
file(GLOB SRC_FILES src/*.cpp)

file(GLOB FLWR_SRC_FILES ../../src/cc/flwr/src/*.cc)
file(GLOB FLWR_INCLUDE_FILES ../../src/cc/flwr/include/*)
set(FLWR_INCLUDE_DIR ../../src/cc/flwr/include)
set(PROTO_FILES transport.proto)

add_library(flwr STATIC
  ${FLWR_SRC_FILES}
  ${FLWR_INCLUDE_FILES}
  ${PROTO_FILES})
target_link_libraries(flwr
  protobuf::libprotobuf
  grpc++)
target_include_directories(flwr PUBLIC
  proto
  ${CMAKE_CURRENT_BINARY_DIR}
  ${FLWR_INCLUDE_DIR})
protobuf_generate(TARGET flwr LANGUAGE cpp)
cmake_policy(SET CMP0026 OLD)
get_target_property(grpc_cpp_plugin_location grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET flwr LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")

add_executable(${EXECUTABLE_NAME}
  ${SRC_FILES})
target_include_directories(${EXECUTABLE_NAME} PUBLIC
  include
  ${FLWR_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE_NAME}
  TorchVision::TorchVision
  flwr)
