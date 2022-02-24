cmake_minimum_required(VERSION 3.16)
project(SimpleCppFlowerClient VERSION 0.10
  DESCRIPTION "Creates a Simple C++ Flower client that trains a linear model on synthetic data."
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(ABSL_PROPAGATE_CXX_STD ON)

######################
### Download gRPC 

include(FetchContent)
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.43.2
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(gRPC)

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()


######################
### FLWR_GRPC_PROTO

get_filename_component(FLWR_PROTO "../../src/proto/flwr/proto/transport.proto" ABSOLUTE)
get_filename_component(FLWR_PROTO_PATH "${FLWR_PROTO}" PATH)

set(FLWR_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/transport.pb.cc")
set(FLWR_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/transport.pb.h")
set(FLWR_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/transport.grpc.pb.cc")
set(FLAR_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/transport.grpc.pb.h")

# External building command to generate gRPC source files.
add_custom_command(
  OUTPUT "${FLWR_PROTO_SRCS}" "${FLWR_PROTO_HDRS}" "${FLWR_GRPC_SRCS}" "${FLWR_GRPC_HDRS}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
	-I "${FLWR_PROTO_PATH}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${FLWR_PROTO}"
  DEPENDS "${FLWR_PROTO}"
)

add_library(flwr_grpc_proto 
  ${FLWR_GRPC_SRCS}
  ${FLWR_GRPC_HDRS}
  ${FLWR_PROTO_SRCS}
  ${FLWR_PROTO_HDRS}
)

target_include_directories(flwr_grpc_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(flwr_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
)

######################
### FLWR_LIB

file(GLOB FLWR_SRCS "../../src/cc/flwr/src/*.cc")
set(FLWR_INCLUDE_DIR "../../src/cc/flwr/include")

add_library(flwr ${FLWR_SRCS})

target_include_directories(flwr PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${FLWR_INCLUDE_DIR}
)

target_link_libraries(flwr
  flwr_grpc_proto
)

######################
### FLWR_CLIENT
file(GLOB FLWR_CLIENT_SRCS src/*.cc)
set(EXECUTABLE_NAME flwr_client)
add_executable(${EXECUTABLE_NAME} ${FLWR_CLIENT_SRCS})

target_include_directories(${EXECUTABLE_NAME} PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  ${FLWR_INCLUDE_DIR}
)

target_link_libraries(${EXECUTABLE_NAME}
  flwr
)
