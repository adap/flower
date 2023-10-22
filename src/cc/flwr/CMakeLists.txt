cmake_minimum_required(VERSION 3.16)
project(flwr VERSION 1.0
  DESCRIPTION "Flower Library that packages gRPC and other dependencies"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(ABSL_PROPAGATE_CXX_STD ON)

# Assume gRPC and other dependencies are necessary
include(FetchContent)
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.43.2
)
FetchContent_MakeAvailable(gRPC)

# Variables for gRPC and Protocol Buffers
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

# FLWR_GRPC_PROTO

get_filename_component(FLWR_PROTO "../../proto/flwr/proto/transport.proto" ABSOLUTE)
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

add_library(flwr_grpc_proto STATIC 
  ${FLWR_GRPC_SRCS}
  ${FLWR_GRPC_HDRS}
  ${FLWR_PROTO_SRCS}
  ${FLWR_PROTO_HDRS}
)

target_include_directories(flwr_grpc_proto 
   PUBLIC 
       $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
   PRIVATE
       ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(flwr_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
)
# For the internal use of flwr
file(GLOB FLWR_SRCS "src/*.cc")

add_library(flwr ${FLWR_SRCS})

target_include_directories(flwr PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Link gRPC and other dependencies
target_link_libraries(flwr PRIVATE flwr_grpc_proto)

# Merge the two libraries
add_library(flwr_merged STATIC $<TARGET_OBJECTS:flwr> $<TARGET_OBJECTS:flwr_grpc_proto>)

target_include_directories(flwr_merged PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# This will create a 'flwrConfig.cmake' for users to find
install(TARGETS flwr_merged EXPORT flwrTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include
)
install(
    FILES 
    ${CMAKE_CURRENT_BINARY_DIR}/transport.grpc.pb.h 
    ${CMAKE_CURRENT_BINARY_DIR}/transport.pb.h
    DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

install(EXPORT flwrTargets
  FILE flwrConfig.cmake
  NAMESPACE flwr::
  DESTINATION lib/cmake/flwr
)

# Optional: Generate and install package version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/flwrConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/flwrConfigVersion.cmake"
  DESTINATION lib/cmake/flwr
)

