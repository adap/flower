cmake_minimum_required(VERSION 3.16)
project(SimpleCppFlowerClient VERSION 0.10
  DESCRIPTION "Creates a Simple C++ Flower client that trains a linear model on synthetic data."
  LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
# set(ABSL_PROPAGATE_CXX_STD ON)

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

FetchContent_Declare(
  flwr_repo
  GIT_REPOSITORY https://github.com/adap/flower.git
  GIT_TAG        add-cpp-proto  # or specify a specific commit or tag if needed
)

FetchContent_GetProperties(flwr_repo)
if(NOT flwr_repo_POPULATED)
  FetchContent_Populate(flwr_repo)
endif()

set(FLWR_SOURCE_ROOT "${flwr_repo_SOURCE_DIR}")
set(FLWR_SDK_PATH "${FLWR_SOURCE_ROOT}/src/cc/flwr")

# FLWR_GRPC_PROTO
# get_filename_component(FLWR_PROTO_BASE_PATH "${FLWR_SOURCE_ROOT}/src/proto/" ABSOLUTE)
# get_filename_component(FLWR_TRANS_PROTO "${FLWR_SOURCE_ROOT}/src/proto/flwr/proto/transport.proto" ABSOLUTE)
# get_filename_component(FLWR_NODE_PROTO "${FLWR_SOURCE_ROOT}/src/proto/flwr/proto/node.proto" ABSOLUTE)
# get_filename_component(FLWR_TASK_PROTO "${FLWR_SOURCE_ROOT}/src/proto/flwr/proto/task.proto" ABSOLUTE)
# get_filename_component(FLWR_FLEET_PROTO "${FLWR_SOURCE_ROOT}/src/proto/flwr/proto/fleet.proto" ABSOLUTE)

# set(FLWR_TRANS_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/transport.pb.cc")
# set(FLWR_TRANS_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/transport.pb.h")
# set(FLWR_TRANS_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/transport.grpc.pb.cc")
# set(FLWR_TRANS_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/transport.grpc.pb.h")

# set(FLWR_NODE_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/node.pb.cc")
# set(FLWR_NODE_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/node.pb.h")
# set(FLWR_NODE_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/node.grpc.pb.cc")
# set(FLWR_NODE_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/node.grpc.pb.h")

# set(FLWR_TASK_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/task.pb.cc")
# set(FLWR_TASK_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/task.pb.h")
# set(FLWR_TASK_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/task.grpc.pb.cc")
# set(FLWR_TASK_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/task.grpc.pb.h")

# set(FLWR_FLEET_PROTO_SRCS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/fleet.pb.cc")
# set(FLWR_FLEET_PROTO_HDRS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/fleet.pb.h")
# set(FLWR_FLEET_GRPC_SRCS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/fleet.grpc.pb.cc")
# set(FLWR_FLEET_GRPC_HDRS "${CMAKE_CURRENT_BINARY_DIR}/flwr/proto/fleet.grpc.pb.h")

# # External building command to generate gRPC source files.
# add_custom_command(
#   OUTPUT "${FLWR_TRANS_PROTO_SRCS}" 
#          "${FLWR_TRANS_PROTO_HDRS}" 
#          "${FLWR_TRANS_GRPC_SRCS}" 
#          "${FLWR_TRANS_GRPC_HDRS}"
#   COMMAND ${_PROTOBUF_PROTOC}
#   ARGS  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
#         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
#         -I "${FLWR_PROTO_BASE_PATH}"
#         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
#         "${FLWR_TRANS_PROTO}"
#         DEPENDS "${FLWR_TRANS_PROTO}"
# )

# add_custom_command(
#   OUTPUT 
#          "${FLWR_NODE_PROTO_SRCS}" 
#          "${FLWR_NODE_PROTO_HDRS}" 
#          "${FLWR_NODE_GRPC_SRCS}" 
#          "${FLWR_NODE_GRPC_HDRS}"
#   COMMAND ${_PROTOBUF_PROTOC}
#   ARGS  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
#         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
#         -I "${FLWR_PROTO_BASE_PATH}"
#         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
#         "${FLWR_NODE_PROTO}"
#         DEPENDS 
#                 "${FLWR_NODE_PROTO}"
# )

# add_custom_command(
#   OUTPUT 
#          "${FLWR_TASK_PROTO_SRCS}" 
#          "${FLWR_TASK_PROTO_HDRS}" 
#          "${FLWR_TASK_GRPC_SRCS}" 
#          "${FLWR_TASK_GRPC_HDRS}"
#   COMMAND ${_PROTOBUF_PROTOC}
#   ARGS  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
#         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
#         -I "${FLWR_PROTO_BASE_PATH}"
#         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
#         "${FLWR_TASK_PROTO}"
#         DEPENDS 
#                 "${FLWR_TASK_PROTO}"
# )

# add_custom_command(
#   OUTPUT 
#          "${FLWR_FLEET_PROTO_SRCS}" 
#          "${FLWR_FLEET_PROTO_HDRS}" 
#          "${FLWR_FLEET_GRPC_SRCS}" 
#          "${FLWR_FLEET_GRPC_HDRS}"
#   COMMAND ${_PROTOBUF_PROTOC}
#   ARGS  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
#         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
#         -I "${FLWR_PROTO_BASE_PATH}"
#         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
#         "${FLWR_FLEET_PROTO}"
#         DEPENDS 
#                 "${FLWR_FLEET_PROTO}"
# )

# add_library(flwr_grpc_proto STATIC 
#   ${FLWR_TRANS_GRPC_SRCS}
#   ${FLWR_TRANS_GRPC_HDRS}
#   ${FLWR_TRANS_PROTO_SRCS}
#   ${FLWR_TRANS_PROTO_HDRS}
#   ${FLWR_NODE_GRPC_SRCS}
#   ${FLWR_NODE_GRPC_HDRS}
#   ${FLWR_NODE_PROTO_SRCS}
#   ${FLWR_NODE_PROTO_HDRS}
#   ${FLWR_TASK_GRPC_SRCS}
#   ${FLWR_TASK_GRPC_HDRS}
#   ${FLWR_TASK_PROTO_SRCS}
#   ${FLWR_TASK_PROTO_HDRS}
#   ${FLWR_FLEET_GRPC_SRCS}
#   ${FLWR_FLEET_GRPC_HDRS}
#   ${FLWR_FLEET_PROTO_SRCS}
#   ${FLWR_FLEET_PROTO_HDRS}
# )

# target_include_directories(flwr_grpc_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# target_link_libraries(flwr_grpc_proto
#   ${_REFLECTION}
#   ${_GRPC_GRPCPP}
#   ${_PROTOBUF_LIBPROTOBUF}
# )

######################
### FLWR_LIB

file(GLOB FLWR_SRCS "${FLWR_SDK_PATH}/src/*.cc")
set(FLWR_INCLUDE_DIR "${FLWR_SDK_PATH}/include")

add_library(flwr ${FLWR_SRCS})

target_include_directories(flwr PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}
  ${FLWR_INCLUDE_DIR}
)

target_link_libraries(flwr
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
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
